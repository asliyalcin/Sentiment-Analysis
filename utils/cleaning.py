import re 

def clean_text(text,stopwords_tr_nltk):
    text = text.lower()
    text = re.sub(r'[^a-zçğıöşü\s]', '', text)
    words = [w for w in text.split() if w not in stopwords_tr_nltk]
    return ' '.join(words)


def normalize_for_embedding(text):
    if not isinstance(text, str):
        return ""
    text = " ".join(text.split())
    return text
