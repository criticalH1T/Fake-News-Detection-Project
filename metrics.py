import numpy as np


def confusion_matrix_and_accuracy(true_labels, predictions):
    # Initialize true positives, true negatives, false positives, and false negatives to zero
    tp, tn, fp, fn = 0, 0, 0, 0

    # Iterate over each true label and corresponding prediction
    for i, (true_label, prediction) in enumerate(zip(true_labels, predictions)):
        # Print the true label and prediction for the first 10 examples
        if i < 10:
            print(f"True label: {true_label}, Prediction: {prediction}")

        # Increment the appropriate variable based on whether the prediction was correct or not
        if true_label == 'real' and prediction == 1:
            tp += 1
        elif true_label == 'real' and prediction == -1:
            fn += 1
        elif true_label == 'fake' and prediction == 1:
            fp += 1
        elif true_label == 'fake' and prediction == -1:
            tn += 1

    # Create a numpy array representing the confusion matrix
    confusion_matrix = np.array([[tp, fp], [fn, tn]])

    # Calculate the accuracy
    accuracy = calculate_accuracy_from_confusion_matrix(tp, fp, fn, tn)

    # Return the confusion matrix and accuracy
    return confusion_matrix, accuracy


def calculate_accuracy_from_confusion_matrix(tp, fp, fn, tn):
    # Calculate the accuracy based on the true positives, true negatives, false positives, and false negatives
    if (tp + tn + fp + fn) != 0:
        accuracy = (tp + tn) / (tp + tn + fp + fn)
    else:
        return "Cannot calculate accuracy, the denominator is zero."

    # Return the accuracy
    return accuracy


def calculate_precision(tp, fp):
    # Calculate the precision based on the true positives and false positives
    if (tp + fp) != 0:
        precision = tp / (tp + fp)
    else:
        return "Cannot calculate precision, the denominator is zero."

    # Return the precision
    return precision


def calculate_recall(tp, fn):
    # Calculate the recall based on the true positives and false negatives
    if (tp + fn) != 0:
        recall = tp / (tp + fn)
    else:
        return "Cannot calculate recall, the denominator is zero."

    # Return the recall
    return recall


def calculate_f1_score(precision, recall):
    # Calculate the F1 score based on the precision and recall
    if (precision + recall) != 0:
        f1_score = 2 * ((precision * recall) / (precision + recall))
    else:
        return "Cannot calculate F1 score, the denominator is zero."

    # Return the F1 score
    return f1_score
