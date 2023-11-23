# Import necessary modules
from flask import Flask, request, jsonify
from flask_cors import CORS
from functools import wraps
from Passive_Aggressive_Classifier import PassiveAggressiveClassifier
from TF_IDF import TFIDFVectorizer
from preprocessing import preprocess_text, train_test_split
from reading_writing import read_preprocessed_csv
from metrics import *

# Initialize Flask app and enable cross-origin resource sharing (CORS)
app = Flask(__name__)
CORS(app)


# Define a decorator for authentication
def auth_required(f):
    # Define the decorated function
    @wraps(f)
    def decorated(*args, **kwargs):
        # Check if the authorization header is present and if the username and password match
        auth = request.authorization
        if auth and auth.username == 'user' and auth.password == 'password':
            # If authentication is successful, execute the function
            return f(*args, **kwargs)
        else:
            # If authentication fails, return an error message and a 401 status code
            return jsonify({'message': 'Authentication failed'}), 401

    return decorated


# Load the preprocessed data from a CSV file
all_data = read_preprocessed_csv("assets/cleaned_dataset.csv")

# Split the data into training and test sets
X_train, y_train, X_test, y_test = train_test_split(all_data[:30000])

# Load the trained classifier and vectorizer
classifier = PassiveAggressiveClassifier()
vectorizer = TFIDFVectorizer(n_features=2 ** 17)
X_train_vectorized = vectorizer.fit_transform(X_train)
classifier.fit(X_train_vectorized, y_train)

X_test_vectorized = vectorizer.transform(X_test)

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


# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
@auth_required
def predict():
    # Get the text data from the request
    data = request.json

    # Preprocess the text data
    preprocessed_text = preprocess_text(data)

    # Vectorize the text data
    vectorized_text = vectorizer.transform([preprocessed_text])

    # Make predictions
    prediction = classifier.predict(vectorized_text)

    # Return the predictions as a JSON response
    return jsonify({'prediction': prediction[0]})


if __name__ == '__main__':
    # Start the Flask app on port 5000
    app.run(port=5000)
