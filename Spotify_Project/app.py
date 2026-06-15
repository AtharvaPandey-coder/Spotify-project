# Importing all the necessary things
import streamlit as st
import pandas as pd
import time
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
from vaderSentiment.vaderSentiment import (
    SentimentIntensityAnalyzer
)
from mood_detector import mood_detect_vader

st.set_page_config(page_title='Spotify Recommender',
                   layout='wide')
st.title('Music Recommendation System')
with st.sidebar:
    st.info('this Model Predicts Song')
    st.success('Tech USed-> Cosine Similarity + TFIDF Vectorizer + Vader')
    st.success('It is Build By Atharva')

df=pickle.load(open('cleaned_df.pkl','rb'))
tfidf_vectorizer=pickle.load(open('TfidfVectorizer.pkl','rb'))
tfidf_matrix=pickle.load(open('Tfidf_matrix.pkl','rb'))
search_matrix=pickle.load(open('search_matrix.pkl','rb'))
search_vectorizer=pickle.load(open('search_vectorizer.pkl','rb'))


audio_features=['danceability','energy','valence','tempo','loudness','acousticness','liveness','instrumentalness','speechiness']

## helper function
def find_song(song_name):
    song_name=str(song_name).strip()
    mask=df['track_name'].str.lower() == song_name.lower()
    if mask.any():
        idx=df[mask].index[0]
        return df.loc[[idx]],[],None
    
    #3 Partial Match
    mask=df['track_name'].str.lower().str.contains(
        song_name.lower(), regex=False
    )
    if mask.any():
        idx=df[mask].index[0]
        return df.loc[[idx]],[],None
    
    #3 Fuzzy Match
    matches=get_close_matches(song_name,df['track_name'].tolist(),n=3,cutoff=0.6)
    if matches:
        return None,matches,f"song name{song_name} not found"
    return None,[],f"{song_name} not found"


with st.sidebar:
    st.header('Dataset Info')
    st.metric('Total Songs',len(df))
    st.metric('Total Artist',df['track_artist'].nunique())



#3 Tabs Section 
tab1,tab2=st.tabs([
    'Song Recommender',
    "Mood(Vader)",
])


#3 Cosine similarity
with tab1:
    st.header('Song recommendation')
    st.write(
        "Find Songs based on Audio Features"
        "by using **Cosine Similarity ** "
    )
    song_input=st.text_area(
        "Enter song Name",placeholder='Believer',key="t1"
    )
    top_n_1=st.slider(
        'Number of recommendation You Want',
        3,5,10,key='n1'
    )

    if st.button('Find Similar Song'):
        if song_input.strip():
            song,sug,error=find_song(song_input)

            if sug and error:
                st.warning('err')
                st.info(
                    "Did You Mean :"
                    f"**{', '.join(sug)}**"
                )
            elif error:
                st.error(error)
            else:
                song_idx=song.index[0]
                song_vec=df.loc[[song_idx],audio_features].to_numpy(dtype=np.float64)
                all_vec=df[audio_features].to_numpy(dtype=np.float64)
                sims=cosine_similarity(song_vec,all_vec)[0]
                top_idx=np.argsort(sims)[-top_n_1-1:-1][::-1]
                results=df.iloc[top_idx][[
                    'track_name','track_artist','valence','energy','mood','tempo_cat','danceability'
                ]].copy()
                results['similarity']=sims[top_idx].round(3)
                
                st.success(
                    f"Top  {top_n_1} similar songs"
                )
                st.dataframe(results,use_container_width=True)
        else:
            st.warning('Please Enter a song')

#3 Tab 2 Vader Input
with tab2:
    st.header("Mood based Recommendation")
    st.write('vader detects Your emotion and Return matching Songs')
    st.markdown("""
        "Try Typing",
        "I feel good Today",
        "I Feel Sad Today",
        "I Feel Motivated Today",
    """)
    mood_text=st.text_input('How re You Feeling?',
                            placeholder='I feel Good today',
                            key='t2')
    if st.button('Get Songs',key='btn3'):
        with st.spinner('Detecting Mood..'):
            time.sleep(2)
        results,mood,confidence=mood_detect_vader(mood_text,df)
        st.dataframe(results,use_container_width=True)
    else:
        st.warning('Please enter the Text ABOVE')

























