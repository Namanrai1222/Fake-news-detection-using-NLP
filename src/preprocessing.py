import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r'^\s*[A-Z\s]+\(\s*Reuters\s*\)\s*-\s*', '', text)
    text = re.sub(r'\(Reuters\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Reuters', '', text, flags=re.IGNORECASE)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    text = " ".join(words[15:])
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)