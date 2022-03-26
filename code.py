import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn


import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

import yfinance as yf


stocks = ['GOOG', 'FB', 'AMZN', 'AAPL', 'MSFT', 'SPY']

ohlc = {}


def download_data():

    for stock in stocks:

        ticker = yf.Ticker(stock)
        ohlc[stock] = ticker.history(period='1y', interval='1d')['Close']

    return pd.DataFrame(ohlc)


def find_cointegrated_pairs(data):
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < 0.05:
                pairs.append((keys[i], keys[j]))
    return score_matrix, pvalue_matrix, pairs


def zscore(series):
    return (series - series.mean()) / np.std(series)


if __name__ == '__main__':
    df = download_data()

    prices_df = df.copy()
    prices_df.dropna(axis=0, how='all', inplace=True)
    prices_df.dropna(axis=0, how='any', inplace=True)

    scores, pvalues, pairs = find_cointegrated_pairs(prices_df)
    seaborn.heatmap(pvalues, xticklabels=stocks, yticklabels=stocks,
                    cmap='RdYlGn_r', mask=(pvalues >= 0.05))
    print(pairs)

    S1 = prices_df[pairs[0][0]]
    S2 = prices_df[pairs[0][1]]
    score, pvalue, _ = coint(S1, S2)
    print(pvalue)

    S1 = sm.add_constant(S1)
    results = sm.OLS(S2, S1).fit()
    S1 = S1[pairs[0][0]]
    b = results.params[pairs[0][0]]

    spread = S2 - b * S1
    spread.plot()
    plt.axhline(spread.mean(), color='black')
    plt.legend(['Spread'])

    zscore(spread).plot()
    plt.axhline(zscore(spread).mean(), color='black')
    plt.axhline(1.0, color='red', linestyle='--')
    plt.axhline(-1.0, color='green', linestyle='--')
    plt.legend(['Spread z-score', 'Mean', '+1', '-1'])

    df_strategy = pd.DataFrame()
    df_strategy['spread'] = spread
    df_strategy['z score'] = zscore(spread)
    df_strategy['long'] = np.where(df_strategy['z score'] < -1.0, 1, 0)
    df_strategy['short'] = np.where(df_strategy['z score'] > 1.0, 1, 0)
    df_strategy['exit'] = np.where(df_strategy['z score'] == 0.0, 1, 0)

    df_strategy['logReturn'] = np.log(-df_strategy['spread']).diff()
    df_strategy['shiftedReturn'] = df_strategy['logReturn'].shift(-1)

    df_strategy.loc[:, 'algoReturn'] = 0

    algoreturn = 0
    for row in df.index:

        if df_strategy['long'][row]:
            algoreturn = -1 * df_strategy['shiftedReturn'][row]

        elif df_strategy['short'][row]:
            algoreturn = 1 * df_strategy['shiftedReturn'][row]

        else:
            algoreturn = 0

        df_strategy.loc[row, 'algoReturn'] = algoreturn

    df_strategy['algoReturn'].plot()
    plt.show()
    df_strategy['algoReturn'].dropna(how='any', inplace=True)
    print(df_strategy['algoReturn'])
    print(df_strategy['algoReturn'].sum())
