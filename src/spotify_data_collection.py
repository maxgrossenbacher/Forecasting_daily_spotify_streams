import os
import subprocess
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# reading in data
df = pd.read_csv('../data/data.csv')
df.dropna(axis=0, inplace=True)
# spliting URL and creating song id column
df[['https', 'none', 'spotify', 'track','id']] =  df['URL'].str.split('/', expand=True)
# drop unnecessary cols
df.drop(labels=['https', 'none', 'spotify', 'track'], axis=1,inplace=True)
print(df.head())
#song ids to set
track_ids = set(df['id'].tolist())
audio_features = []
ids_scraped = []
d = []
for i in track_ids:
    # creating urls to scrape Spotify API
    url = "https://api.spotify.com/v1/audio-features/"+i
    print(url)
    #calling cURL script
    audio_features.append(subprocess.check_output(["/Users/gmgtex/Desktop/projects/kaggle_spotify/src/spotify_api_new.sh" , url], universal_newlines=True))
for t in audio_features:
    if i not in ids_scraped:
        ids_scraped.append(i)
        r = t.split(',\n')
        arr = [k.split(' : ') for k in r]
        if arr == [['{\n  "error"', '{\n    "status"', '404'],['    "message"', '"analysis not found"\n  }\n}']]:
            break
        if dict(arr) not in d:
            d.append(dict(arr))
# creating audio features attributes dataframe from ids in df
df1 = pd.DataFrame(d)
#formating and merging columns on track id
df1.rename(columns={'  "acousticness"':"acousticness",'  "analysis_url"':"analysis_url", '  "duration_ms"':"duration_ms", \
                    '  "energy"': "energy", '  "id"':"id", '  "instrumentalness"': "instrumentalness", '  "key"':"key", \
                    '  "liveness"': "liveness",'  "loudness"': "loudness", '  "mode"': "mode", '  "speechiness"': "speechiness", \
                    '  "tempo"': "tempo",'  "time_signature"': "time_signature", '  "track_href"': "track_href", '  "type"': "type", \
                    '  "uri"': "uri",'  "valence"': "valence", '{\n  "danceability"': "dancability"}, inplace=True)
df1['id']=df1.id.str.strip("\"")
# join dfs on song id
spotify_df = df.merge(df1, on='id', how='left', suffixes=['_track', '_af'], sort=False, copy=False, indicator=True)
# spotify_df.to_pickle('spotify_audio_features.pkl')
