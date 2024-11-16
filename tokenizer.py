import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt_tab')
stemmer = PorterStemmer()

def tokenize(content):
    token_list = []
    cur_token = ""

    for char in content:
        if char.isalnum() and char.isascii():
            cur_token += char
        else:
            if cur_token:
                token_list.append(cur_token.lower())
                cur_token = ""
    if cur_token:
        token_list.append(cur_token.lower())

    stemmed_tokens = [stemmer.stem(token) for token in token_list]

    return stemmed_tokens