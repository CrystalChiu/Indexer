import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt_tab')
stemmer = PorterStemmer()

def tokenize(content):
    # FIXME: this is just a stand in for now to test other mechanisms
    tokens = word_tokenize(content.lower())

    # get rid of non alphanum characters
    filtered_tokens = [re.sub(r'\W+', '', token) for token in tokens if token.isalnum()]

    # use porter stemming on acquired tokens
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]

    return stemmed_tokens
