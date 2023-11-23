import numpy as np


# Define the Passive Aggressive Classifier class
class PassiveAggressiveClassifier:
    # Initialize the class
    def __init__(self, C=1.0, max_iter=1000, tol=1e-3):
        # Set the regularization strength
        self.C = C
        # Set the maximum number of iterations
        self.max_iter = max_iter
        # Set the tolerance for early stopping
        self.tol = tol
        # Set the weight vector to None
        self.w = None

    # Define the fit method
    def fit(self, X, y):
        # Get the number of samples and features
        n_samples, n_features = X.shape
        # Initialize the weight vector to zeros
        self.w = np.zeros(n_features)
        # Define a mapping from labels to integers
        label_to_int = {'real': 1, 'fake': -1}

        # Loop through the number of iterations
        for _ in range(self.max_iter):
            # Loop through each sample
            for i in range(n_samples):
                # Get the sample and label
                xi, yi = X[i], y[i]
                # Convert the label to an integer
                yi = label_to_int[yi]
                # Calculate the margin
                margin = yi * np.dot(np.array(xi), self.w)
                # If the margin is less than 1, update the weight vector
                if margin < 1:
                    update = (1 - margin) / (np.dot(xi, xi) + 1 / (2 * self.C))
                    self.w += update * yi * xi
            # Check if the average margin is greater than 1 minus the tolerance
            if np.mean(margin) > 1 - self.tol:
                break

    # Define the predict method
    def predict(self, X):
        # Make predictions using the dot product of X and the weight vector
        predictions = np.sign(np.dot(X, self.w))
        # Replace zeros with ones in the predictions
        return np.where(predictions == 0, 1, predictions)

    # Define the get_feature_importance method
    def get_feature_importance(self):
        # Get the absolute weight values
        abs_weights = np.abs(self.w)
        # Normalize the weight values
        norm_weights = abs_weights / np.sum(abs_weights)
        # Get the feature importance ranking
        ranking = np.argsort(norm_weights)[::-1]
        # Return the ranking
        return ranking

    # Define the save_weights method
    def save_weights(self, filepath):
        with open(filepath, 'w') as f:
            # Write the weight vector to a file as a comma-separated string
            f.write(','.join(str(weight) for weight in self.w))

    # Define the load_weights method
    def load_weights(self, filepath):
        with open(filepath, 'r') as f:
            # Read the weight vector from a file as a comma-separated string
            weights_str = f.readline().strip().split(',')
            # Convert the weight string to a numpy array
            self.w = np.array([float(weight) for weight in weights_str])
