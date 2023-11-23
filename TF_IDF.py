import json
import numpy as np


class TFIDFVectorizer:
    def __init__(self, n_features=2 ** 16):
        self.n_features = n_features
        self.word_to_index = {}  # dictionary to map words to hashed indices
        self.idf = {}  # dictionary to store IDF values for each word

    # Compute the term frequency (TF) for a list of words
    @staticmethod
    def _compute_tf(words):
        term_count = {}
        for word in words:
            term_count[word] = term_count.get(word, 0) + 1

        total_words = len(words)
        tf = {}
        for word, count in term_count.items():
            tf[word] = count / total_words
        return tf

    # Compute the inverse document frequency (IDF) for each word in the corpus
    def _compute_idf(self, data):
        document_count = len(data)
        word_document_count = np.zeros(self.n_features)

        for article in data:
            words = set(article['title'] + article['text'])
            for word in words:
                hashed_index = hash(word) % self.n_features
                if hashed_index not in self.word_to_index:
                    self.word_to_index[hashed_index] = word
                word_document_count[hashed_index] += 1

        idf = {}
        for word_index, count in enumerate(word_document_count):
            if count > 0:
                idf[word_index] = np.log(document_count / count)
        return idf

    # Compute the TF-IDF vectors for all articles in the corpus
    def fit_transform(self, data):
        idf = self._compute_idf(data)
        self.idf = idf
        tf_idf_vectors = []

        for article in data:
            words = article['title'] + article['text']
            tf = self._compute_tf(words)

            tf_idf_vector = np.zeros(self.n_features)
            for word, term_frequency in tf.items():
                hashed_index = hash(word) % self.n_features
                if hashed_index in idf:
                    tf_idf_vector[hashed_index] = term_frequency * idf[hashed_index]
            tf_idf_vectors.append(tf_idf_vector)

        return np.array(tf_idf_vectors)

    # Compute the TF-IDF vectors for new data using the precomputed IDF values
    def transform(self, data):
        tf_idf_vectors = []

        for article in data:
            words = article['text']
            tf = self._compute_tf(words)

            tf_idf_vector = np.zeros(self.n_features)
            for word, term_frequency in tf.items():
                hashed_index = hash(word) % self.n_features
                tf_idf_vector[hashed_index] = term_frequency * self.idf.get(hashed_index, 0)
            tf_idf_vectors.append(tf_idf_vector)

        return np.array(tf_idf_vectors)

    # Save the state of the vectorizer (n_features, word_to_index, idf) to a JSON file
    def save_state(self, filepath):
        state_dict = {
            'n_features': self.n_features,
            'word_to_index': self.word_to_index,
            'idf': self.idf
        }
        with open(filepath, 'w') as f:
            f.write(json.dumps(state_dict))

    # Load the state of the vectorizer(n_features, word_to_index, idf) to the vectorizer object
    def load_state(self, filepath):
        with open(filepath, 'r') as f:
            state_dict = json.load(f)
        self.n_features = state_dict['n_features']
        self.word_to_index = state_dict['word_to_index']
        self.idf = state_dict['idf']
