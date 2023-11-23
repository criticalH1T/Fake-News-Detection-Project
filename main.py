from Passive_Aggressive_Classifier import PassiveAggressiveClassifier
from metrics import *
from preprocessing import *
from TF_IDF import TFIDFVectorizer


def main():
    # Load data
    data = read_preprocessed_csv("assets/cleaned_dataset.csv")

    # Split into training and test sets
    X_train, y_train, X_test, y_test = train_test_split(data[:30000])

    # Vectorize training data
    vectorizer = TFIDFVectorizer(n_features=2 ** 16)
    X_train_vectorized = vectorizer.fit_transform(X_train)

    # Train classifier
    classifier = PassiveAggressiveClassifier()
    classifier.fit(X_train_vectorized, y_train)

    # classifier.save_weights('demo_assets/weights.txt')

    # Vectorize test data
    # vectorizer.save_state('demo_assets/vectorizer.json')
    X_test_vectorized = vectorizer.transform(X_test)

    # Make predictions
    predictions = classifier.predict(X_test_vectorized)

    confusion_matrix, accuracy = confusion_matrix_and_accuracy(y_test, predictions)

    # Calculate precision
    precision = calculate_precision(confusion_matrix[0][0], confusion_matrix[0][1])

    # Calculate recall
    recall = calculate_recall(confusion_matrix[0][0], confusion_matrix[1][0])

    # Calculate F1 score
    f1_score = calculate_f1_score(precision, recall)

    print(f"Confusion matrix:\n{confusion_matrix}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 score: {f1_score:.4f}")


if __name__ == "__main__":
    main()
