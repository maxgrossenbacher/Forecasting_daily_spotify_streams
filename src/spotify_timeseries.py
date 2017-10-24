import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
from spotify_data_crawler import Collector
import os
from datetime import datetime, timedelta, date
import csv
import threading
import requests
plt.style.use('ggplot')


def bigger_fonts(ax):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)

def get_top_songs(df):
    top_o_charts = df[df['Position'] == 1]
    songs_that_topped = set((top_o_charts['Track Name'].tolist()))
    region_charts = top_o_charts[['Date','Region','Streams','Artist','Track Name', 'day_of_week','day']].reset_index()
    return region_charts

def plot_1_song(df, title, col):
    fig, ax = plt.subplots(1)
    for track in set(df['Track Name'].tolist()):
        df[df['Track Name'] == track].plot(ax=ax, x='Date', \
                                                y=col, figsize=(20,15),\
                                                title=title,\
                                                color=np.random.rand(3,1).T)
        plt.ylabel('Total # Streams')
        plt.legend(set(df['Track Name'].tolist()))
        bigger_fonts(ax)


def test_stationarity(timeseries, std=True):

    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=7)
    rolstd = pd.rolling_std(timeseries, window=7)

    #Plot rolling statistics:
    fig, ax = plt.subplots()
    orig = ax.plot(timeseries, color='orange',label='Original')
    mean = ax.plot(rolmean, color='black', label='Rolling Mean')
    if std:
        std = ax.plot(rolstd, color='red', label = 'Rolling Std')
    fig.set_size_inches((20,10))
    plt.xlabel('Date')
    plt.ylabel('Log Avg # Streams')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    bigger_fonts(ax)

    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    return rolmean

def seasonal_decomp(ts):
    from statsmodels.tsa.seasonal import seasonal_decompose
    decomposition = seasonal_decompose(x=ts, freq=12)

    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    fig,ax = plt.subplots(4)
    ax1,ax2,ax3,ax4 = ax

    ax1.plot(ts, label='Original', c='orange')
    ax1.legend(loc='best')

    ax2.plot(trend, label='Trend', c='blue')
    ax2.legend(loc='best')

    ax3.plot(seasonal,label='Seasonality', c='purple')
    ax3.legend(loc='best')

    ax4.plot(residual, label='Residuals',c ='red')
    ax4.legend(loc='best')

    fig.set_size_inches(30,20)
    bigger_fonts(ax1)
    bigger_fonts(ax2)
    bigger_fonts(ax3)
    bigger_fonts(ax4)
    return residual


def ARIMA_model(df, p, q):
    model = ARIMA(df, order=(p, 1, q))
    results_ARIMA = model.fit(disp=0)
    predictions_ARIMA = results_ARIMA.predict()
    return model, predictions_ARIMA

def transforming_predictions(col_log, ARIMA_predictions):
    predictions_ARIMA_log = pd.Series(col_log.iloc[0], index=col_log.index)
    predictions_ARIMA_log = predictions_ARIMA_log.add(ARIMA_predictions.cumsum(),fill_value=0)
    predictions_ARIMA = np.exp(predictions_ARIMA_log)
    return predictions_ARIMA

if __name__ == '__main__':
    DATA_DIRECTORY = 'data'
    one_day = timedelta(days=1)
    start_date = date(2017, 1, 1)
    end_date = datetime.now().date() - (2 * one_day)

    regions = ["global", "us"]

    for region in regions:
        collector = Collector(region, start_date, end_date)
        collector.start()

    spotify_df = pd.read_csv('data/global.csv', infer_datetime_format=True, parse_dates=['Date'])
    spotify_df['day_of_week'] = spotify_df['Date'].dt.dayofweek
    spotify_df['day'] = spotify_df['day_of_week'].map({0:'Monday',1:'Tuesday',\
                                                     2:'Wednesday',3:'Thursday',\
                                                     4:'Friday',5:'Saturday',6:'Sunday'})
    top_songs_df = get_top_songs(spotify_df)
    plot_1_song(top_songs_df, 'Global Streams of #1 Song', col='Streams')
