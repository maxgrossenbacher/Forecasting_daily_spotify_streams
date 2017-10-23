# Forecasting Number of Spotify Streams using Time Series

## Code
* [EDA and Forecasting Notebook](https://github.com/maxgrossenbacher/kaggle_spotify/blob/master/src/Spotify_Time_Series.ipynb) - In this notebook, I forecast the avg. number of streams and the number of streams for the #1 song on Spotify using ARIMA models.
* [Getting Audio Features](https://github.com/maxgrossenbacher/kaggle_spotify/blob/master/src/spotify_data_collection.py) - script for using Spotify API to get audio features of song on Spotify  

## Conclusions & Future Directions
We can see that ARIMA models are able to capture the weekly trends in number of daily Spotify streams. We can see that number of daily streams are lowest on Sundays and grow gradually throughout the week, peaking on Fridays.
![alt_text](../images/Avg_Streaming_Jan_2017.png)
![alt_text](../images/Avg_Streaming_Day_of_Week.png)

However, the ARIMA models not very good at forecasting monthly trends of daily number of Spotify streams. Since, this data is only for an 8 month period, the ARIMA models are having a tough time forecasting the monthly underlying signal. A longer period of data would allow the ARIMA models to better capture this signal.  

Some future directions for this project would be to try LSTM neural net to improve the forecasting number of daily streams on Spotify.

## Tech Stack
* Jupyter Notebook
* Pandas
* Numpy
* Matplotlib
* Statsmodels
* TimeSeries - ARIMA
