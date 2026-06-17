from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
lemmatizer=WordNetLemmatizer() 
from nltk.corpus import stopwords
stop_words=set(stopwords.words('english'))
import re
def clean_text(text):
    text = str(text).lower()
    # remove urls
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # remove mentions and hashtags (keeps hashtag words if you prefer remove the # only use: text = re.sub(r'#','',text))
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    # keep only letters and spaces
    text = re.sub(r'[^a-z\s]', ' ', text)
    # collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # tokenize
    tokens = word_tokenize(text)
    # remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)