import re


def word_tokenize(text):
    """
    Tokenizes a given text into words.
    """
    clean_text = re.sub(r'[^\w\s]', '', text)
    clean_text = re.sub(r'[*\d+]', '', clean_text)
    words = clean_text[:-12].split(' ')
    return words


def load_stopwords():
    """
    Loads stopwords from a file and returns them as a list.
    """
    with open("assets/stopwords.txt") as f:
        stop_words = str(f.readlines())[2:-2].split(',')
        return stop_words


stopwords = load_stopwords()


def remove_stopwords(text: list):
    """
    Removes stopwords from a given list of words.
    """
    text = [x.lower() for x in text]
    result = filter(lambda x: x not in stopwords and x != '', text)
    return list(result)


def remove_non_ascii(text: list):
    """
    Removes non-ascii characters from a given list of words.
    """
    for word in text:
        if not word.isascii():
            text.remove(word)
    return text


def preprocess(text):
    """
    Performs preprocessing on a given text by tokenizing, removing non-ascii characters, and removing stopwords.
    """
    words = word_tokenize(text)
    words = remove_non_ascii(words)
    stopwords_removed = remove_stopwords(words)
    return stopwords_removed
