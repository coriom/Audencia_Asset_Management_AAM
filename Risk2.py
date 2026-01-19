# ============================================================
# Imports (garde seulement ceux dont tu as besoin)
# ============================================================
import io
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ============================================================
# 1) Utilitaires Portefeuille
# ============================================================
def build_shares_timeseries(trades: pd.DataFrame,
                            prices_index: pd.DatetimeIndex,
                            tickers: list[str]) -> pd.DataFrame:
    """
    Construit une matrice 'quantités détenues' (index = dates de marché, colonnes = tickers)
    à partir d'un DataFrame de transactions:
      colonnes requises: ['ticker','date','qty']
      - qty > 0 : achat, qty < 0 : vente
    La quantité est tenue en cumul (ffill) dès la date d'effet de la transaction.
    """
    idx = pd.DatetimeIndex(prices_index).unique().sort_values()
    shares = pd.DataFrame(0.0, index=idx, columns=tickers, dtype=float)

    for tkr, df_t in trades.groupby('ticker'):
        if tkr not in shares.columns:
            continue
        df_t = df_t.sort_values('date')
        for _, row in df_t.iterrows():
            d = pd.to_datetime(row['date'])
            pos = shares.index.searchsorted(d)
            if pos >= len(shares.index):
                continue
            d_effective = shares.index[pos]
            shares.loc[d_effective:, tkr] += float(row['qty'])

    return shares.astype(float)


def portfolio_analytics(close_prices: pd.DataFrame,
                        shares: pd.DataFrame) -> dict:
    """
    Calcule la valeur du portefeuille, rendements, courbe base 100 et drawdown.
    - close_prices: DataFrame des prix 'Close' (index = dates, colonnes = tickers)
    - shares: DataFrame des quantités (mêmes index/colonnes)
    """
    close_prices = close_prices.reindex(shares.index).reindex(columns=shares.columns)
    values_by_asset = (close_prices * shares)
    port_value = values_by_asset.sum(axis=1)

    port_returns = port_value.pct_change().fillna(0.0)
    equity_100 = 100 * (1 + port_returns).cumprod()

    roll_max = port_value.cummax()
    drawdown = (port_value - roll_max) / roll_max
    max_dd = float(drawdown.min())

    return {
        "values_by_asset": values_by_asset,
        "portfolio_value": port_value,
        "portfolio_returns": port_returns,
        "equity_100": equity_100,
        "drawdown": drawdown,
        "max_drawdown": max_dd,
    }


# ============================================================
# 2) Normalisation base 10 et tracés par actif
# ============================================================
def normalize_prices_base(close_prices: pd.DataFrame,
                          base: float = 10.0,
                          base_date_per_ticker: dict | None = None) -> pd.DataFrame:
    """
    Normalise chaque série de prix à 'base' (par défaut 10) à une date de référence.
    - close_prices : DataFrame (index=dates, colonnes=tickers), prix 'Close'
    - base : niveau de normalisation (10 par défaut)
    - base_date_per_ticker : dict {ticker: 'YYYY-MM-DD'} (sinon 1re date de la série)
    """
    norm = pd.DataFrame(index=close_prices.index, columns=close_prices.columns, dtype=float)
    for t in close_prices.columns:
        s = close_prices[t].dropna()
        if s.empty:
            continue

        if base_date_per_ticker and t in base_date_per_ticker:
            d0 = pd.to_datetime(base_date_per_ticker[t])
            pos = s.index.searchsorted(d0)
            base_price = s.iloc[pos] if pos < len(s) else s.iloc[0]
        else:
            base_price = s.iloc[0]

        norm[t] = base * (s / base_price)

    return norm


def plot_prices_per_asset(close_prices: pd.DataFrame,
                          base: float = 10.0,
                          base_date_per_ticker: dict | None = None,
                          use_normalized: bool = True,
                          suptitle: str | None = None,
                          figsize=(10, 4)):
    """
    Trace un graphique par actif (une figure par ticker).
    - close_prices : DataFrame (index=dates, colonnes=tickers)
    - base : base de normalisation (10 par défaut) si use_normalized=True
    - base_date_per_ticker : dict {ticker: 'YYYY-MM-DD'} pour la base par actif
    - use_normalized : True -> on trace les prix normalisés à 'base'; False -> prix bruts
    - suptitle : suffixe de titre facultatif
    """
    if use_normalized:
        data_to_plot = normalize_prices_base(close_prices, base=base,
                                             base_date_per_ticker=base_date_per_ticker)
        title_prefix = f"Prix normalisés (base={base})"
    else:
        data_to_plot = close_prices
        title_prefix = "Prix"

    for t in data_to_plot.columns:
        s = data_to_plot[t].dropna()
        if s.empty:
            continue

        plt.figure(figsize=figsize)
        s.plot()
        if use_normalized and base_date_per_ticker and t in base_date_per_ticker:
            tbase = pd.to_datetime(base_date_per_ticker[t])
            if s.index.min() <= tbase <= s.index.max():
                plt.axvline(tbase, linestyle="--", alpha=0.5)

        full_title = f"{title_prefix} – {t}"
        if suptitle:
            full_title += f" | {suptitle}"

        plt.title(full_title)
        plt.xlabel("Date")
        plt.ylabel("Niveau" if use_normalized else "Prix")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# ============================================================
# 3) Tracker portefeuille (sans stooq_download ici)
#    -> suppose que stooq_download(tickers, start, end) est dispo
# ============================================================
def tracker_portefeuille(trades_input,
                         start_date=None,
                         end_date=None,
                         plot=True,
                         plot_per_asset=True,
                         normalize_base=10.0):
    """
    Suivi de portefeuille à partir d'ordres datés.
    - trades_input: list[dict] ou DataFrame colonnes: 'ticker','date','qty'
         • 'ticker' : symbole Stooq (ex: 'aapl.us', '^spx')
         • 'date'   : 'YYYY-MM-DD'
         • 'qty'    : achat >0 / vente <0
    - start_date/end_date: bornes d'analyse (auto si None)
    - plot: tracer courbe portefeuille + contributions + drawdown
    - plot_per_asset: tracer aussi un graphique par actif (base = normalize_base)
    - normalize_base: base de normalisation pour les tracés par actif (10 par défaut)
    """
    # 0) Ingest
    trades = pd.DataFrame(trades_input).copy()
    trades['date'] = pd.to_datetime(trades['date'])
    tickers = sorted(trades['ticker'].unique().tolist())

    # 1) Fenêtre d'analyse
    if start_date is None:
        start_date = (trades['date'].min() - pd.Timedelta(days=7)).date().isoformat()
    if end_date is None:
        end_date = pd.Timestamp.utcnow().date().isoformat()

    # 2) Prix via Stooq (à fournir/Importer à l'extérieur)
    px = stooq_download(tickers, start=start_date, end=end_date)
    if isinstance(px.columns, pd.MultiIndex):
        close = px.xs('Close', axis=1, level=1)
    else:
        if 'Close' not in px.columns:
            raise KeyError("Pas de colonne 'Close' dans les prix récupérés.")
        close = px[['Close']]
        close.columns = [tickers[0]]

    # 3) Quantités détenues
    shares = build_shares_timeseries(trades, close.index, tickers)

    # 4) Analytics
    ana = portfolio_analytics(close, shares)

    # 5) Contributions journalières par actif
    values_by_asset = ana['values_by_asset']
    contributions = values_by_asset.div(values_by_asset.sum(axis=1), axis=0).fillna(0.0)

    # 6) Plots portefeuille
    if plot:
        # Courbe base 100
        plt.figure(figsize=(12,5))
        ana['equity_100'].plot(title="Courbe de portefeuille (base 100)")
        plt.xlabel("Date"); plt.ylabel("Base 100")
        plt.grid(True); plt.tight_layout(); plt.show()

        # Contributions empilées
        plt.figure(figsize=(12,5))
        contributions.plot.area(ax=plt.gca(), title="Contributions (%) par actif", ylim=(0,1))
        plt.xlabel("Date"); plt.ylabel("Part du portefeuille")
        plt.tight_layout(); plt.show()

        # Drawdown
        plt.figure(figsize=(12,3.5))
        ana['drawdown'].plot(title=f"Drawdown du portefeuille (max: {ana['max_drawdown']:.2%})",
                             color="red")
        plt.xlabel("Date"); plt.ylabel("Drawdown")
        plt.grid(True); plt.tight_layout(); plt.show()

    # 7) Plots par actif (base = 10 par défaut)
    if plot_per_asset:
        base_dates = trades.groupby('ticker')['date'].min().astype(str).to_dict()
        plot_prices_per_asset(close,
                              base=normalize_base,
                              base_date_per_ticker=base_dates,
                              use_normalized=True,
                              suptitle="(base = date 1re entrée)")

    return {
        "close": close,
        "shares": shares,
        "values_by_asset": values_by_asset,
        "portfolio_value": ana['portfolio_value'],
        "portfolio_returns": ana['portfolio_returns'],
        "equity_100": ana['equity_100'],
        "drawdown": ana['drawdown'],
        "max_drawdown": ana['max_drawdown'],
        "contributions": contributions,
        "trades": trades.sort_values('date')
    }


# ============================================================
# 4) Fonctions “analyse” (optionnelles) déjà adaptées Stooq
#    — utiles si tu veux réutiliser au même endroit
# ============================================================
def afficher_matrice_correlation(tickers, start_date, end_date):
    data = stooq_download(tickers, start=start_date, end=end_date)
    if isinstance(data.columns, pd.MultiIndex):
        close_prices = data.xs('Close', axis=1, level=1)
    else:
        if 'Close' not in data.columns:
            raise KeyError("Les données téléchargées ne contiennent pas de colonne 'Close'.")
        close_prices = data[['Close']]
        close_prices.columns = [tickers[0]] if isinstance(tickers, (list, tuple)) else [str(tickers)]

    returns = close_prices.pct_change().dropna(how='all').dropna(axis=1, how='all')
    if returns.empty or returns.shape[1] < 1:
        raise ValueError("Pas assez de données pour calculer la corrélation.")

    corr = returns.corr()
    plt.figure(figsize=(10, 8))
    import seaborn as sns
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title("Corrélation des rendements journaliers (Stooq - Close)")
    plt.tight_layout()
    plt.show()


def afficher_matrice_covariance(tickers, start_date, end_date):
    data = stooq_download(tickers, start=start_date, end=end_date)
    if isinstance(data.columns, pd.MultiIndex):
        close_prices = data.xs('Close', axis=1, level=1)
    else:
        if 'Close' not in data.columns:
            raise KeyError("Les données téléchargées ne contiennent pas 'Close'.")
        close_prices = data[['Close']]
        close_prices.columns = [tickers[0]] if isinstance(tickers, (list, tuple)) else [str(tickers)]

    returns = close_prices.pct_change().dropna(how="all")
    cov = returns.cov()
    plt.figure(figsize=(10, 8))
    import seaborn as sns
    sns.heatmap(cov, annot=True, fmt=".6f", cmap="viridis")
    plt.title("Covariance des rendements journaliers (Stooq - Close)")
    plt.tight_layout()
    plt.show()


def calculer_var(tickers, start_date, end_date, confidence_level, initial_investment, z_score):
    data = stooq_download(tickers, start=start_date, end=end_date)
    if isinstance(data.columns, pd.MultiIndex):
        close_prices = data.xs('Close', axis=1, level=1)
    else:
        if 'Close' not in data.columns:
            raise KeyError("Pas de 'Close' dans les données.")
        close_prices = data[['Close']]
        close_prices.columns = [tickers[0]] if isinstance(tickers, (list, tuple)) else [str(tickers)]

    returns = close_prices.pct_change().dropna(how='all').dropna(axis=1, how='all')
    if returns.empty:
        raise ValueError("Pas assez de données de rendements pour la VaR.")

    historical_var = -returns.quantile(1 - confidence_level)
    parametric_var = z_score * returns.std() - returns.mean()

    var_df = pd.DataFrame({
        'VaR Historique ($)': historical_var * initial_investment,
        'VaR Paramétrique ($)': parametric_var * initial_investment
    }).sort_index()

    print(f"=== Value at Risk à {int(confidence_level * 100)}% de confiance ===")
    print(var_df.round(2))

    ax = var_df.plot(kind='bar', figsize=(10, 6),
                     title=f"VaR par actif (portefeuille de {initial_investment:,.0f} $)")
    plt.ylabel("Pertes potentielles ($)")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
    return var_df


def calculer_volatilite(indice, start_date, end_date):
    data = stooq_download(indice, start=start_date, end=end_date)
    if isinstance(data.columns, pd.MultiIndex):
        close_prices = data.xs('Close', axis=1, level=1)
        close = close_prices.iloc[:, 0]
    else:
        if 'Close' not in data.columns:
            raise KeyError("Pas de 'Close'.")
        close = data['Close']

    log_returns = np.log(close / close.shift(1))
    returns = log_returns.dropna()
    if returns.empty:
        raise ValueError(f"Aucune donnée exploitable pour {indice}.")

    volatility = returns.std() * np.sqrt(252)
    print(f"Volatilité annualisée de {indice} entre {start_date} et {end_date} : {volatility:.2%}")

    returns.plot(title=f"Rendements Logarithmiques Quotidiens - {indice}")
    plt.xlabel("Date"); plt.ylabel("Log returns")
    plt.grid(True); plt.tight_layout(); plt.show()
    return float(volatility)


def calculer_betas(tickers, indice, start_date, end_date):
    if isinstance(tickers, str):
        tickers = [tickers]
    symbols = list(tickers) + [indice]
    data = stooq_download(symbols, start=start_date, end=end_date)

    if isinstance(data.columns, pd.MultiIndex):
        close = data.xs('Close', axis=1, level=1)
    else:
        if 'Close' not in data.columns:
            raise KeyError("Pas de 'Close'.")
        close = data[['Close']]
        close.columns = [symbols[0]]

    returns = close.pct_change().dropna(how='any')
    if indice not in returns.columns:
        raise KeyError(f"L'indice '{indice}' est manquant.")
    var_mkt = returns[indice].var(ddof=1)
    if var_mkt == 0:
        raise ValueError("Variance de l'indice nulle (bêta indéterminé).")

    betas = {}
    for t in tickers:
        if t not in returns.columns:
            betas[t] = np.nan
            continue
        cov_im = returns[[t, indice]].cov().iloc[0, 1]
        betas[t] = cov_im / var_mkt if var_mkt != 0 else np.nan

    df_betas = pd.DataFrame.from_dict(betas, orient='index', columns=['Bêta']).sort_index()
    print(df_betas)
    return df_betas


def calculer_sharpe_ratio(tickers, start_date, end_date, risk_free_rate):
    if isinstance(tickers, str):
        tickers = [tickers]
    data = stooq_download(tickers, start=start_date, end=end_date)

    if isinstance(data.columns, pd.MultiIndex):
        prices = data.xs('Close', axis=1, level=1)
    else:
        if 'Close' not in data.columns:
            raise KeyError("Pas de 'Close'.")
        prices = data[['Close']]
        prices.columns = [tickers[0]]

    returns = prices.pct_change().dropna(how="all")
    if returns.empty:
        raise ValueError("Pas assez de données pour le Sharpe.")

    mean_returns = returns.mean() * 252
    volatility = returns.std() * np.sqrt(252)
    sharpe = (mean_returns - risk_free_rate) / volatility

    df_sharpe = pd.DataFrame({
        'Rendement annualisé': mean_returns,
        'Volatilité annualisée': volatility,
        'Sharpe Ratio': sharpe
    }).round(4)

    print(f"=== Ratio de Sharpe (taux sans risque = {risk_free_rate*100:.2f}%) ===")
    print(df_sharpe)
    return df_sharpe


def corr_moyenne_simple(tickers, start_date, end_date):
    if isinstance(tickers, str):
        tickers = [tickers]
    data = stooq_download(tickers, start=start_date, end=end_date)

    if isinstance(data.columns, pd.MultiIndex):
        close_prices = data.xs('Close', axis=1, level=1)
    else:
        if 'Close' not in data.columns:
            raise KeyError("Pas de 'Close'.")
        close_prices = data[['Close']]
        close_prices.columns = [tickers[0]]

    returns = close_prices.pct_change().dropna(how="all")
    if returns.empty or returns.shape[1] < 2:
        raise ValueError("Il faut au moins 2 tickers valides.")

    corr_matrix = returns.corr()
    mask = ~np.eye(len(corr_matrix), dtype=bool)
    mean_corr = corr_matrix.where(mask).mean().mean()
    print(f"Corrélation moyenne simple du portefeuille : {mean_corr:.4f}")
    return float(mean_corr)


def corr_moyenne_ponderee(tickers, start_date, end_date, weights=None):
    if isinstance(tickers, str):
        tickers = [tickers]
    data = stooq_download(tickers, start=start_date, end=end_date)

    if isinstance(data.columns, pd.MultiIndex):
        close_prices = data.xs('Close', axis=1, level=1)
    else:
        if 'Close' not in data.columns:
            raise KeyError("Pas de 'Close'.")
        close_prices = data[['Close']]
        close_prices.columns = [tickers[0]]

    returns = close_prices.pct_change().dropna(how="all")
    if returns.empty or returns.shape[1] < 2:
        raise ValueError("Il faut au moins 2 tickers valides.")

    corr_matrix = returns.corr()
    n = len(tickers)

    if weights is None:
        w = np.ones(n) / n
    else:
        if isinstance(weights, dict):
            w = np.array([weights.get(t, 0) for t in tickers])
        else:
            w = np.array(weights)
        w = w / w.sum()

    total_weight = 0.0
    weighted_sum = 0.0
    # Suppose que l'ordre des colonnes correspond à 'tickers'
    cols = corr_matrix.columns.tolist()
    for i in range(n):
        for j in range(i+1, n):
            if tickers[i] not in cols or tickers[j] not in cols:
                continue
            weight_pair = w[i] * w[j]
            weighted_sum += corr_matrix.loc[tickers[i], tickers[j]] * weight_pair
            total_weight += weight_pair

    mean_corr_weighted = weighted_sum / total_weight if total_weight > 0 else np.nan
    print(f"Corrélation moyenne pondérée du portefeuille : {mean_corr_weighted:.4f}")
    return float(mean_corr_weighted)


# ============================================================
# 5) Exemple d'utilisation local (désactive si tu importes comme module)
# ============================================================
if __name__ == "__main__":
    # ⚠️ Assure-toi que stooq_download est importé/défini ailleurs avant d'exécuter.
# Mapping (nom -> ticker Stooq estimé)
# NVIDIA                       -> nvda.us
# Air Liquide (Euronext Paris) -> ai.fr
# NextEra Energy (US)          -> nee.us
# La Française des Jeux (FR)   -> fdj.fr
# Reliance Industries (GDR LSE)-> rigd.uk   # (à vérifier sur Stooq)
# Gaztransport & Technigaz     -> gtt.fr

    TRADES_EXEMPLE = [
        {"ticker": "nvda.us", "date": "2024-02-15", "qty": 10},
        {"ticker": "ai.fr",   "date": "2024-03-01", "qty": 20},
        {"ticker": "nee.us",  "date": "2024-04-01", "qty": 15},
        {"ticker": "fdj.fr",  "date": "2024-05-10", "qty": 30},
        {"ticker": "rigd.uk", "date": "2024-06-01", "qty": 25},  # Reliance GDR à Londres (à confirmer)
        {"ticker": "gtt.fr",  "date": "2024-07-15", "qty": 40},
    ]
    # Exemple tracker (si stooq_download dispo):
    try:
        res = tracker_portefeuille(TRADES_EXEMPLE, start_date="2021-12-01",
                                   end_date=None, plot=True, plot_per_asset=True, normalize_base=10.0)
        print("\nMax Drawdown:", f"{res['max_drawdown']:.2%}")
    except NameError:
        print("➡️ Définis / importe d'abord stooq_download(tickers, start, end).")
