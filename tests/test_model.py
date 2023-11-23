import unittest
import numpy as np
from Passive_Aggressive_Classifier import PassiveAggressiveClassifier


class TestPassiveAggressiveClassifier(unittest.TestCase):
    def setUp(self):
        self.clf = PassiveAggressiveClassifier()

    def test_fit_predict(self):
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        y_train = np.array(['real', 'fake', 'real'])
        X_test = np.array([[1, 1], [2, 2], [5, 5]])

        # Ensure the model can fit the training data without errors
        self.clf.fit(X_train, y_train)

        # Ensure the model can make predictions on the test data without errors
        y_pred = self.clf.predict(X_test)
        self.assertEqual(len(y_pred), len(X_test))

    def test_save_load_weights(self):
        # Ensure the model can save and load weights without errors
        self.clf.w = np.array([1, 2, 3])
        self.clf.save_weights('test_weights.txt')
        self.clf.load_weights('test_weights.txt')
        self.assertTrue(np.array_equal(self.clf.w, np.array([1, 2, 3])))
