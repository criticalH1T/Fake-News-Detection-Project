import csv
import numpy as np


def save_word_corpus_to_csv(word_to_index, file_path):
    """
    saves word corpus to csv
    """
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Word', 'Index'])
        for word, index in word_to_index.items():
            writer.writerow([word, index])


def save_tf_idf_vectors_to_csv(tf_idf_vectors, index_to_word, labels, file_path):
    """
    saves tf idf vectors to csv
    """
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        header = ['Label'] + [index_to_word[i] for i in range(len(index_to_word))]
        writer.writerow(header)

        for label, vector in zip(labels, tf_idf_vectors):
            row = [label] + vector.tolist()
            writer.writerow(row)


def load_word_corpus_from_csv(file_path):
    """
    loads word corpus from csv
    """
    word_to_index = {}
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            word, index = row
            word_to_index[word] = int(index)
    return word_to_index


def load_tf_idf_vectors_from_csv(file_path):
    """
    loads tf idf vectors from csv
    """
    tf_idf_vectors = []
    labels = []
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Skip header
        for row in reader:
            label, *vector = row
            labels.append(label)
            tf_idf_vectors.append([float(value) for value in vector])
    return np.array(tf_idf_vectors), labels


def save_idf_to_csv(idf, file_path):
    """
    saves idfs to csv
    """
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['word', 'idf'])
        for word, value in idf.items():
            writer.writerow([word, value])


def load_idf_from_csv(file_path):
    """
    loads idfs to csv
    """
    idf = {}
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            word, value = row
            idf[word] = float(value)
    return idf
