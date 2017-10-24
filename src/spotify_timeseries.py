import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
from sklearn.metrics import mean_squared_error
plt.style.use('ggplot')


def bigger_fonts(ax, fsize=17):
    '''
    Function for making font sizes on figures bigger
    '''
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fsize)

def get_top_songs(df):
    '''
    Function for returning #1 ranked songs on Spotify (total number streams)
    '''
    top_o_charts = df[df['Position'] == 1]
    songs_that_topped = set((top_o_charts['Track Name'].tolist()))
    region_charts = top_o_charts[['Date','Region','Streams','Artist','Track Name', 'day_of_week','day']].reset_index()
    return region_charts

def plot_1_song(df, title, col):
    '''
    Function for ploting #streams per track. Each track is its own unique color
    '''
    fig, ax = plt.subplots(1)
    for track in set(df['Track Name'].tolist()):
        df[df['Track Name'] == track].plot(ax=ax, x='Date', \
                                                y=col, figsize=(20,15),\
                                                title=title,\
                                                color=np.random.rand(3,1).T)
        plt.ylabel('Total # Streams')
        plt.legend(set(df['Track Name'].tolist()))
        bigger_fonts(ax)

def make_timeseries(df):
    return df.set_index('Date')

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
    bigger_fonts(ax, 15)

    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries.iloc[:,0].values, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    return rolmean

def seasonal_decomp(ts):
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


def ARIMA_model(df, p, d, q):
    model = ARIMA(df, order=(p, d, q))
    results_ARIMA = model.fit(disp=0)
    predictions_ARIMA = results_ARIMA.predict()
    return model, predictions_ARIMA

def transforming_predictions(ts_log, ARIMA_predictions):
    predictions_ARIMA_log = pd.Series(float(ts_log.iloc[0]), index=ARIMA_predictions.index)
    predictions_ARIMA_log = predictions_ARIMA_log.add(ARIMA_predictions.cumsum(),fill_value=0)
    predictions_ARIMA = np.exp(predictions_ARIMA_log)
    return predictions_ARIMA

if __name__ == '__main__':

    print('loading data...')
    global_df = pd.read_csv('data/global.csv', infer_datetime_format=True, parse_dates=['Date'])
    global_df['day_of_week'] = global_df['Date'].dt.dayofweek
    global_df['day'] = global_df['day_of_week'].map({0:'Monday',1:'Tuesday',\
                                                     2:'Wednesday',3:'Thursday',\
                                                     4:'Friday',5:'Saturday',6:'Sunday'})
    # get top song Global and # streams
    top_songs_global_df = get_top_songs(global_df)
    # plot number one song Global and # streams
    plot_1_song(top_songs_global_df, 'Global Streams of #1 Song', col='Streams')

    print('loading data...')
    us_df = pd.read_csv('data/us.csv', infer_datetime_format=True, parse_dates=['Date'])
    us_df['day_of_week'] = us_df['Date'].dt.dayofweek
    us_df['day'] = us_df['day_of_week'].map({0:'Monday',1:'Tuesday',\
                                                     2:'Wednesday',3:'Thursday',\
                                                     4:'Friday',5:'Saturday',6:'Sunday'})
    # get top song US and # streams
    top_us_songs_df = get_top_songs(us_df)
    # plot number one song US and # streams
    plot_1_song(top_us_songs_df, 'US Streams of #1 Song', col='Streams')
    plt.show()

    # creating ARIMA model
    total_streams = global_df.groupby('Date').sum()["Streams"].reset_index()
    global_ts = make_timeseries(total_streams)
    global_ts_log = np.log(global_ts)
    rol_mean = test_stationarity(global_ts_log, std=False)
    plt.show()

    model, global_predictions_ARIMA = ARIMA_model(global_ts_log, 2, 1, 4)
    trans_preds_ARIMA = transforming_predictions(global_ts_log, global_predictions_ARIMA)
    trans_preds_ARIMA.dropna(inplace=True)

    fig, ax = plt.subplots(1)
    ax.plot(global_ts, label='Total Streams')
    ax.plot(trans_preds_ARIMA , label='ARIMA prediction')
    plt.title('ARIMA model to Predict Average Number of Streams RMSE = {}' .format(np.sqrt(mean_squared_error(global_ts[1:], trans_preds_ARIMA))))
    plt.xlabel('Date')
    plt.ylabel('Total # Streams')
    plt.legend(loc='best')
    bigger_fonts(ax, 15)
    fig.set_size_inches((20,10))
    print('RMSE: {}'.format(np.sqrt(mean_squared_error(global_ts[1:], trans_preds_ARIMA))))
    plt.show()
