import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Liste des tickers boursiers (exemples : Apple, Amazon, Google, Microsoft)
tickers = ['AAPL', 'AMZN', 'GOOGL', 'MSFT']

# Téléchargement des données sur les 6 derniers mois
data = yf.download(tickers, start='2023-10-01', end='2024-04-01')['Adj Close']

# Calcul du rendement journalier
returns = data.pct_change().dropna()

# Création de la matrice de corrélation
correlation_matrix = returns.corr()

# Affichage
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Matrice de corrélation des rendements')
plt.show()
