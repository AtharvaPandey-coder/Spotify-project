import streamlit as st
import numpy as np
import pandas as pd
import pickle
from text_cleaning import clean_text
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title='Mental health Analyzer',layout='wide',initial_sidebar_state='expanded')
MAX_LEN=150


WELLNESS_TIPS={
    'Normal':{
        "message":"You seen to be in a good Mental Space Kepp Nuturing Your Welbeing",
        "tips":[
            "Keep up your physical activities routine-> it boost mood naturally",
            "Practice mindfullness or meditation daily",
            "read Something inspiring and learn a new skill"
        ]
    },

    "Depression":{
        "message":"It sounds like you are going through a really tough time.You're not alone.",
        "tips":[
            "Read [Bhagwad-Gita] for 1 Hour it make you feel Good",
            "Try to get sunlight in the morning it refresher your brain",
            "Talk with your frieds or spent time with family"
        ]
    },
    'Anxiety':{
        'message':'Anxiety can feel overwhelming ,but there are ways to manage it effictively',
        'tips':[
            "Try box breathing inhale 4s-> hold 4s -> exhale 4s -> hold 4s",
            "Reduce News and Social Media consumption",
            "Write Down your worries it helps externalize your thoughts"
        ]
    },

    'Bpolar':{
        'message':'bipolar Disorder is manageable with the right support and consistency',
        'tips':[
            'Pritiorize sleep consistency -> irregular sleep can trigger episodes',
            'Build a support network of people who support you',
            'track your mood daily with a journal or a app',
        ]
    },
    'Stress':{
        'message':'stress is your mind signal that needs attention',
        'tips':[
            'make a priority list and tackle one thing at a time',
            'Take regular breakes--> use pomodoro technique',
            'learn to say no'
        ]
    },
    'Personality Disorder':{
        'message':'Living with a personality disorder is challenging',
        'tips':[
            'Kepp a emotion journal to keep track of your emotion',
            'Build a stable support Network ',
            'Focus on a one small positive change at a time'
        ]
    },
    'Suicidal':{
        'message':'Please reach out for help immediately -> your life has value',
        'tips':[
            'Go to your nearest Hospital emergency department',
            'Tell someone you trust about how you are feeling',
            'You are not a burden.People do care for you'
        ]
    },
}

EXAMPLES ={
    'Depression':'I Feel completely empy .Nothing Brings me joy anymore',
    'Anxiety':'I Keep worrying about everything all the time.',
    'Normal':'Today was a great day! I went for a walk,met some friends',
    'Stress':'Work has been overwhelming lately.'
}

Class_INFO={
    'Depression':'Persistent sadness,loss of interest,lasting 2+ weeks',
    'Anxiety':'Excessive worry,restlessness,physical symptoms like racing heart',
    'Bipolar':'Extreme mood swings and depressive lows',
    'Stress':'Emotional/Physical tension from external pressures',
    'Personality Disorder':'Enduring patterns afferting relationshps and self-image',
    'Sucidal':'Thought of ending ones life',
    'Normal':'No concern mental health detected from your text'
}

## Now loading the model
@st.cache_resource
def load_artifact():
    model = load_model('bidirectional_lstm.h5')
    tokenizer = pickle.load(open('Tokenizer.pkl','rb'))
#df=pickle.load(open('cleaned_df.pkl','rb'))
    le=pickle.load(open('LabelEncoder.pkl','rb'))
    return model,tokenizer,le

def predict(text,model,tokenizer,le):
    cleaned=clean_text(text)
    seq=tokenizer.texts_to_sequences([cleaned])
    padded=pad_sequences(seq,maxlen=MAX_LEN,padding='post',truncating='post')
    probs=model.predict(padded,verbose=0)[0]
    idx=np.argmax(probs)
    label=le.inverse_transform([idx])[0]
    confidence=float(probs[idx])
    return label,confidence

with st.sidebar:
    st.header('This ia a mental health Analyzer model')
    st.info('This model is build by using Deep Learning-> BiLSTM')
    st.success('Developed By Atharva')
    st.success('It will also gives wellness tips')

st.title('Mental health text Analyzer')
st.write('Share your feelings and Thoughts')

try:
    model,tokenizer,le=load_artifact()
    model_loaded=True
except Exception as e:
    model_loaded=False
    st.error('Model Not found')

user_text=st.text_area('Hou are you Feeling?',placeholder=' Example:I am feeling Happy Today')
if st.button('Analyze'):
    if not model_loaded:
        st.warning('The Model is not loaded')
    elif not user_text.strip():
        st.warning('Please enter the text Above')
    else:
        label,confidence=predict(user_text,model,tokenizer,le)
        tips_data=WELLNESS_TIPS.get(label,WELLNESS_TIPS['Normal'])
        st.subheader(f"Detected:{label}")
        st.write(f"Confidence:{confidence *100:.1f}%")

        st.markdown("----")
        st.subheader('Wellness tips')
        st.write(tips_data['message'])

        for tip in tips_data['tips']:
            st.write(f"-{tip}")
        if label == 'Suicidal':
            st.error('If You re in Immediate danger please call (112)')













