import pandas as pd
import numpy as np
import os

def load_and_merge_all(data_path):
    # 1. Chargement
    df_games = pd.read_csv(os.path.join(data_path, 'games.csv'))
    df_prices = pd.read_csv(os.path.join(data_path, 'prices.csv'))
    df_players = pd.read_csv(os.path.join(data_path, 'players.csv'), usecols=['playerid', 'country'])
    df_reviews = pd.read_csv(os.path.join(data_path, 'reviews.csv'), usecols=['gameid', 'playerid'], nrows=800000)

    for df in [df_games, df_prices, df_reviews]:
        if 'id' in df.columns: df.rename(columns={'id': 'gameid'}, inplace=True)
        if 'AppID' in df.columns: df.rename(columns={'AppID': 'gameid'}, inplace=True)

    # 2. Fusion et Filtrage Qualité (On ne garde que les jeux avec un minimum d'existence)
    global_hype = df_reviews['gameid'].value_counts().reset_index(name='Global_Hype')
    df_master = pd.merge(df_games, global_hype, on='gameid')
    df_master = pd.merge(df_master, df_prices[['gameid', 'usd']], on='gameid')
    df_master = df_master[df_master['Global_Hype'] > 5].copy()

    # 3. Multi-Hot Encoding des Genres (L'IA voit tous les genres en même temps)
    top_genres = ['Action', 'Adventure', 'Indie', 'RPG', 'Strategy', 'Simulation', 'Casual']
    for g in top_genres:
        df_master[f'genre_{g}'] = df_master['genres'].str.contains(g, na=False).astype(int)

    # 4. Feature Engineering
    df_master['date'] = pd.to_datetime(df_master['release_date'], errors='coerce')
    df_master['Month_Sin'] = np.sin(2 * np.pi * df_master['date'].dt.month / 12)
    df_master['Month_Cos'] = np.cos(2 * np.pi * df_master['date'].dt.month / 12)
    df_master['Year'] = df_master['date'].dt.year
    df_master['is_Multi'] = df_master['supported_languages'].str.count(',').fillna(0)
    df_master['Log_Hype'] = np.log1p(df_master['Global_Hype'])

    # Nettoyage des NaNs
    df_master = df_master.dropna(subset=['usd', 'Log_Hype', 'Month_Sin', 'Year'])

    # 5. Label de Popularité (High/Medium/Low)
    geo_stats = pd.merge(df_reviews, df_players, on='playerid')
    pop_counts = geo_stats.groupby(['gameid', 'country']).size().reset_index(name='score')
    
    def discretize(group):
        if len(group) < 3: group['Popularity'] = 'Low'
        else: group['Popularity'] = pd.qcut(group['score'].rank(method='first'), 3, labels=['Low', 'Medium', 'High'])
        return group
    df_pop = pop_counts.groupby('country', group_keys=False).apply(discretize)

    # 6. Finalisation
    feat_cols = ['Month_Sin', 'Month_Cos', 'Year', 'is_Multi'] + [f'genre_{g}' for g in top_genres]
    df_reg = df_master[feat_cols + ['usd', 'Log_Hype']]
    df_class = pd.merge(df_pop, df_master[feat_cols + ['gameid', 'usd']], on='gameid')
    
    return df_reg, df_class[feat_cols + ['country', 'usd', 'Popularity']]