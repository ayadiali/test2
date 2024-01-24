import subprocess
import sys
import streamlit as st
import random
import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler
import torch
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
# from IPython.display import YouTubeVideo

# from ret import get_genre, get_id_from_info, audio_based, cos_sim 

def get_id_from_info(song, artist, info):
    song_entry = info[(info['song'] == song) & (info['artist'] == artist)]
    
    if not song_entry.empty:
        id = song_entry.iloc[0]['id']
        return id
    
def cos_sim(arr1, arr2):
    arr1_reshape = arr1.reshape(1, -1)
    arr2_reshape = arr2.reshape(1, -1)
    res = cosine_similarity(arr1_reshape, arr2_reshape)[0][0]
    return res

def audio_based(id, dfrepr, N, sim_func):
    # return the query song's row in repr
    target_row = dfrepr[dfrepr['id'] == id].iloc[:, 2:].to_numpy()
    # calculate similarity score
    dfrepr['sim_score'] = dfrepr.apply(lambda x:sim_func(x[2:].to_numpy(),target_row), axis=1)
    # sort tracks by similarity 
    sorted_repr = dfrepr.sort_values(by='sim_score', ascending=False)
    # get the N most similar tracks 
    res = sorted_repr.iloc[1: N+1]['id'].to_numpy()
    return res 

def get_genre(id,genres_df):
  # print(genres_df[genres_df['id'] == id ]['id'].values[0],'--->',id)
  return set(genres_df[genres_df['id'] == id ]['genre'].values[0].replace("[", "").replace("]", "").replace("'", "").split(', '))

# getting the genres of the retrieved ids
#get_genre(id,genres_df)
def id_and_url_or_genre(query_id,retrieved_ids,df,func):
  retrieved_g_u = []
  for id in retrieved_ids:
    retrieved_g_u.append(func(id,df))

  query_g_u = func(query_id,df)
  print(f'Query id: {query_id} ---- Query genre_or_url: {query_g_u}')
  print('---------------------------------------------------------------------------')

  for id, ret in(zip(retrieved_ids,retrieved_g_u)):
    print(f'Retrieved id: {id} ---- Retrieved genre_or_url: {ret}')
  return query_g_u, retrieved_g_u
# query_genre, retrieved_genre = id_and_url_or_genre(query_id=query_id,retrieved_ids=retrieved_ids,df=genre,func=get_genre)


def url(id,url_df):
  return url_df[url_df['id']==id].values[0]
# query_url, retrieved_url = id_and_url_or_genre(query_id=query_id,retrieved_ids=retrieved_ids,df=url_df,func=url)
##################################################################################################

df = pd.read_csv("https://media.githubusercontent.com/media/ayadiali/test/main/id_information_mmsr.tsv", delimiter='\t')
df_artist = df['artist'].values.tolist()
df_song = df['song'].values.tolist()

df['artist_song'] = df['artist'] + ' - ' + df['song']
df_artist_song = df['artist_song'].values.tolist()


# df_artist#.values.tolist()

st.title("Music Retrieval System")

def select_name(names):
    selected_name = st.selectbox(f"Select a query song", names)
    return selected_name

# Select a name using a dropdown menu
selected_artist_song = select_name(df_artist_song)

# Display the selected name
if selected_artist_song:
    selected_artist, selected_song = selected_artist_song.split(' - ')
    st.success(f"Song-Artist: {selected_song}-{selected_artist} ")
    
def select_function(func):
    selected_name = st.selectbox(f"Select a function", func)
    return selected_name
functions = ['Early Fusion_(combinatoin of bert and musicnn)','Cosimilarity: audio-based retrieval system']
selected_func = select_function(functions)

if selected_func:
    st.success(f"Function: {selected_func}")
    
    
# text data
df_bert = pd.read_csv("https://media.githubusercontent.com/media/ayadiali/test/main/id_lyrics_bert_mmsr.tsv", delimiter='\t')
# audio_data
df_ivec1024 = pd.read_csv("https://media.githubusercontent.com/media/ayadiali/test/main/id_ivec1024_mmsr.tsv", delimiter='\t')
df_ivec256 = pd.read_csv("https://media.githubusercontent.com/media/ayadiali/test/main/id_ivec256_mmsr.tsv", delimiter='\t')
df_musicnn = pd.read_csv("https://media.githubusercontent.com/media/ayadiali/test/main/id_musicnn_mmsr.tsv", delimiter='\t')

# url info
url_df = pd.read_csv("https://media.githubusercontent.com/media/ayadiali/test/main/id_url_mmsr.tsv", delimiter='\t')
# genres
genre =  pd.read_csv("https://media.githubusercontent.com/media/ayadiali/test/main/id_genres_mmsr.tsv", delimiter='\t')


### Data preparation

# Merge 2 feature DataFrames based on the 'id' column
merged_df = pd.merge(df_bert, df_musicnn, on='id', how='inner')

# data preprocessing
scaler = StandardScaler()
normalized_features = scaler.fit_transform(merged_df.drop('id', axis=1))
normalized_features_tensor = torch.from_numpy(normalized_features) # transform numpy to tensor
# df_normalized(normalized_features)
df_normalized = pd.concat([merged_df['id'], pd.DataFrame(normalized_features_tensor)], axis=1)

   
if selected_func == 'Early Fusion_(combinatoin of bert and musicnn)':
    query_id = get_id_from_info(song=selected_song, artist=selected_artist, info=df)
    st.markdown(f"**Query song ID:** {query_id}")
                # #     retrieve 10 tracks using combined_normalized data/featuers
    retrieved_ids_norm = audio_based(id=query_id, dfrepr=df_normalized , N=10, sim_func=cos_sim)
    query_url_norm, ret_url_norm = id_and_url_or_genre(query_id=query_id,retrieved_ids=retrieved_ids_norm,df=url_df,func=url)
#     query_genre, retrieved_genre = id_and_url_or_genre(query_id=query_id,retrieved_ids=retrieved_ids_norm,df=genre,func=get_genre)
    query_genre, retrieved_genre = get_genre(id=query_id,genres_df=genre),[get_genre(id=ret_id,genres_df=genre) for ret_id in retrieved_ids_norm]
    
    st.markdown("**Query Ids-Urls:**")
    query_id, query_url = query_url_norm[0], query_url_norm[1]
    st.markdown(f"- <span style='color: #008080;'>**ID:**</span> {query_id}, -----> <span style='color: #008080;'>**URL:**</span> {query_url}",unsafe_allow_html=True)

    st.markdown("**Retrieved Ids-Urls:**")
    for  item in (ret_url_norm):
        ret_id, ret_url = item[0], item[1]
        st.markdown(f"- <span style='color: #008080;'>**ID:**</span> {ret_id}, -----> <span style='color: #008080;'>**URL:**</span> {ret_url}",unsafe_allow_html=True)

    st.markdown("**The Youtube video of the query song:**")
    st.video(query_url_norm[1])


        # selecting a song to play
    st.markdown("**Select a video to play from the retrieved songs**")
    ids, urls = zip(*ret_url_norm) 
    ret_artists_df = df[df['id'].isin(list(ids))]['artist']#.tolist()
    ret_songs_df = df[df['id'].isin(list(ids))]['song']#.tolist()+
    artist_song_ret = pd.DataFrame({'Combined': [f"{artist} - {song}" for artist, song in zip(ret_artists_df, ret_songs_df)]})

# if selected_func == 'Cosimilarity: audio-based retrieval system':
#     query_id = get_id_from_info(song=selected_song, artist=selected_artist, info=df)
#     st.markdown(f"**Query song ID:** {query_id}")
#                 # #     retrieve 10 tracks using combined_normalized data/featuers
#     retrieved_ids_norm = audio_based(id=query_id, dfrepr=df_musicnn , N=10, sim_func=cos_sim)
#     query_url_norm, ret_url_norm = id_and_url_or_genre(query_id=query_id,retrieved_ids=retrieved_ids_norm,df=url_df,func=url)
#     query_genre, retrieved_genre = id_and_url_or_genre(query_id=query_id,retrieved_ids=retrieved_ids_norm,df=genre,func=get_genre)

#     st.markdown("**Query Ids-Urls:**")
#     query_id, query_url = query_url_norm[0], query_url_norm[1]
#     st.markdown(f"- <span style='color: #008080;'>**ID:**</span> {query_id}, -----> <span style='color: #008080;'>**URL:**</span> {query_url}",unsafe_allow_html=True)

#     st.markdown("**Retrieved Ids-Urls:**")
#     for  item in (ret_url_norm):
#         ret_id, ret_url = item[0], item[1]
#         st.markdown(f"- <span style='color: #008080;'>**ID:**</span> {ret_id}, -----> <span style='color: #008080;'>**URL:**</span> {ret_url}",unsafe_allow_html=True)

#         # st.markdown(f"retrieved Urls: {ret_url_norm}")

#     st.markdown("**The Youtube video of the query song:**")
#     st.video(query_url_norm[1])


#         # selecting a song to play
#     st.markdown("**Select a video to play from the retrieved songs**")
#     ids, urls = zip(*ret_url_norm) 
#     ret_artists_df = df[df['id'].isin(list(ids))]['artist']#.tolist()
#     ret_songs_df = df[df['id'].isin(list(ids))]['song']#.tolist()+
#     artist_song_ret = pd.DataFrame({'Combined': [f"{artist} - {song}" for artist, song in zip(ret_artists_df, ret_songs_df)]})    


ret_df_artist_song = artist_song_ret.values.tolist()
# selected_artist_song_ = select_name(artist_song_ret)
selected_artist_song_ = st.selectbox("Select a retrieved track", artist_song_ret)
if selected_artist_song_:
    ret_selected_artist, ret_selected_song = selected_artist_song_.split(' - ')
    st.success(f"Retrieved Song-Retrieved Artist: {ret_selected_song}-{ret_selected_artist}")
    selected_ret_id = get_id_from_info(ret_selected_song, ret_selected_artist, df)
    selected_ret_url = list(urls)[list(ids).index(selected_ret_id)]
    st.markdown("**The Youtube video of the retrieved song :**")
    st.video(selected_ret_url)
    
    # genres
    st.markdown("**Query genres:**")
#     query_genre = query_genre#, query_genre[1]
    st.markdown(f"- <span style='color: #008080;'>**Genre:**</span> {query_genre}", unsafe_allow_html=True)

    st.markdown("**Retrieved genres:**")
    for  track,ret_genre in (zip(ret_df_artist_song,retrieved_genre)):
        st.markdown(f"- <span style='color: #008080;'>**Retrieved track:**</span> {track[0]} -----> **<span style='color: #008080;'>Genre:**</span>{ret_genre}", unsafe_allow_html=True)
