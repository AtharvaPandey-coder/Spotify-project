# 🎵 Spotify Music Recommendation System

A content-based music recommendation system built using
audio feature similarity, lyrical analysis, and
sentiment-based mood detection — deployed as an
interactive Streamlit web application.

---

## 🔍 Project Overview

This project recommends Spotify songs using three
complementary approaches:

- **Audio Similarity** — Cosine similarity on 9 audio
  features (danceability, energy, valence, tempo,
  loudness, acousticness, instrumentalness,
  speechiness, liveness)
- **Lyrics Similarity** — TF-IDF vectorization on
  cleaned song lyrics with cosine similarity to find
  songs sharing similar lyrical themes
- **Mood-Based Recommendation** — VADER sentiment
  analysis detects emotion from user-typed text and
  returns songs matching that mood

---

## 📁 Project Structure
Spotify_Project/
├── data.ipynb # Data cleaning, EDA,
│ # feature engineering
├── model.ipynb # TF-IDF, cosine similarity,
│ # model evaluation
├── app.py # Streamlit application
├── cleaned_df.pkl # Cleaned dataset
├── TfidfVectorizer.pkl # Fitted TF-IDF vectorizer
├── Tfidf_matrix.pkl # Lyrics similarity matrix
├── search_matrix.pkl # Smart search matrix
├── search_vectorizer.pkl # Smart search vectorizer
├── label_encoder.pkl # Mood label encoder
└── requirements.txt # Dependencies


---

## 📊 Dataset

**Source:** [Audio Features and Lyrics of Spotify Songs]
(https://www.kaggle.com/datasets/imuhammad/
audio-features-and-lyrics-of-spotify-songs)

- 18,000+ Spotify songs
- 25 columns including audio features and full lyrics
- Filtered to English-language songs with complete lyrics

---

## 🧹 Data Cleaning & EDA

- Dropped useless ID columns
  (track_id, track_album_id, playlist_id, release_date)
- Removed duplicate songs
  (subset: track_name + track_artist)
- Dropped rows with null or placeholder lyrics
- Filtered lyrics length > 50 characters
- Filtered English-language songs only
- Reset index after all drops

**EDA highlights:**
- Mood distribution across 4 categories
- Correlation heatmap of audio features
- Valence vs Energy scatter plot (mood map)
- Before/after MinMaxScaler comparison

---

## ⚙️ Feature Engineering

| Feature | Description |
|---|---|
| `mood` | Happy/Angry/Calm/Sad from valence + energy |
| `tempo_cat` | Slow/Medium/Fast from tempo BPM |
| `lyrics_length` | Word count of cleaned lyrics |
| `lyrics_clean` | Lowercased, tag-removed, cleaned lyrics |
| `mood_enc` | Label-encoded mood (0–3) |

**Scaling:** MinMaxScaler applied to all 9 audio features
to bring tempo (0–250) and loudness (negative dB) to
the same 0–1 scale as other features — required for
cosine similarity to treat all features equally.

---

## 🤖 Techniques Used

### 1. Cosine Similarity (Audio)
Compares the 9-dimensional audio feature vector of the
input song against every song in the dataset.
Returns top N most similar songs by sound profile.

### 2. TF-IDF + Cosine Similarity (Lyrics)
TF-IDF (max_features=5000, stop_words='english')
converts cleaned lyrics to weighted word vectors.
Cosine similarity finds songs with the most similar
lyrical themes and vocabulary.

### 3. VADER Mood Detection
VADER (Valence Aware Dictionary and sEntiment Reasoner)
analyzes the user's text input and returns a compound
sentiment score mapped to one of 4 mood categories:
Happy, Sad, Angry, or Calm.

### 4. TF-IDF Smart Search
Song descriptions built from mood + tempo + artist
are vectorized. User's natural language query is
transformed by the same vectorizer and matched against
all descriptions using cosine similarity.

### 5. Fuzzy Song Matching
`difflib.get_close_matches` handles user typos.
"Shaep of You" → suggests "Shape of You"

---

## 📈 Key Insight

Mood labels derived from audio features (valence +
energy) do not always align with lyrical sentiment.
For example, "Lose Yourself" by Eminem has high energy
(→ Angry audio label) but lyrics convey motivation and
hope. This mismatch is why VADER was chosen over a
trained NLP classifier for mood input — it operates
directly on user-typed emotional text rather than
song lyrics, avoiding the label-noise problem entirely.

---

## 🖥️ App Features

| Tab | Feature | Method |
|---|---|---|
| 🎵 Song Recommender | Audio similarity | Cosine similarity |
| 📝 Lyrics Similarity | Lyrical theme match | TF-IDF + Cosine |
| 😊 Mood Detection | Text → matching songs | VADER sentiment |
| 🔍 Smart Search | Natural language query | TF-IDF search |
| 📊 EDA Dashboard | Dataset insights | Matplotlib/Seaborn |

---

## 🚀 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 📦 Requirements
pandas
numpy
scikit-learn
streamlit
matplotlib
seaborn
vaderSentiment

## 🏗️ How It Works

User types song name
│
▼
find_song() → exact / partial / fuzzy match
│
▼
Audio features extracted from dataset
│
▼
Cosine similarity vs all 13,691 songs
│
▼
Top N most similar songs returned

User types "I feel sad today"
│
▼
VADER → compound score → mood category
│
▼
Filter df by mood → sample N songs.

