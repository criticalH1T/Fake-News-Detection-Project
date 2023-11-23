from NLP import *
from reading_writing import *
import stemming
import random
import math


def preprocess_text(article: dict):
    # Instantiate a PorterStemmer object
    porter_stemmer = stemming.PorterStemmer()

    # Preprocess the article's text and title using the preprocess function from NLP module
    article['text'] = preprocess(article['text'])

    # Apply stemming to the article's text and title using the PorterStemmer object
    article['text'] = [porter_stemmer.stem(x) for x in article['text']]

    # Return the preprocessed and stemmed article
    return article


def preprocess_data():
    # Read in the fake and real news CSV files using the read_csv function from reading_writing module
    fake = read_csv("assets/fake_news_dataset/Fake.csv")
    real = read_csv("assets/fake_news_dataset/True.csv")

    # Instantiate a PorterStemmer object
    porter_stemmer = stemming.PorterStemmer()

    # Preprocess and stem the data in the fake news CSV file
    for elem in fake:
        if not elem['text'] or not elem['title']:
            fake.remove(elem)
            continue
        elem['title'] = preprocess(elem['title'])
        elem['title'] = [porter_stemmer.stem(x) for x in elem['title']]
        elem['text'] = preprocess(elem['text'])
        elem['text'] = [porter_stemmer.stem(x) for x in elem['text']]
        del elem['subject']
        del elem['date']
        elem['label'] = 'fake'

    # Preprocess and stem the data in the real news CSV file
    for elem in real:
        if not elem['text'] or not elem['title']:
            real.remove(elem)
            continue
        elem['title'] = preprocess(elem['title'])
        elem['title'] = [porter_stemmer.stem(x) for x in elem['title']]
        elem['text'] = preprocess(elem['text'])
        elem['text'] = [porter_stemmer.stem(x) for x in elem['text']]
        del elem['subject']
        del elem['date']
        elem['label'] = 'real'

    # Combine the preprocessed and stemmed data from both CSV files into a single list
    fake.extend(real)

    # Shuffle the data randomly to avoid bias in the training process
    random.shuffle(fake)
    random.shuffle(fake)

    # Write the preprocessed and stemmed data to a CSV file using the write_to_csv function from the reading_writing
    # module
    write_to_csv(fake, 'cleaned_dataset')


# Separate the data into training and testing sets
def train_test_split(data, split_factor=0.8):
    """
    returns training and test lists for dependent columns X and independent column y
    """
    size = len(data)
    n_train = math.floor(split_factor * size)
    training_data = data[:n_train]
    testing_data = data[n_train:]
    X_train = [{'title': row['title'], 'text': row['text']} for row in training_data]
    y_train = [row['label'] for row in training_data]
    X_test = [{'title': row['title'], 'text': row['text']} for row in testing_data]
    y_test = [row['label'] for row in testing_data]
    return X_train, y_train, X_test, y_test
