import unittest
import random
from preprocessing import preprocess, preprocess_text, train_test_split


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        # Create some fake news data
        self.fake_data = [{'title': 'Unbelievable!', 'text': 'Scientists find unicorns in the Amazon!', 'label': 'fake'},
                          {'title': 'Breaking news', 'text': 'Aliens have landed in New York City!', 'label': 'fake'},
                          {'title': 'You won\'t believe this!',
                           'text': 'Elvis Presley is alive and well and living in Mexico!', 'label': 'fake'}]
        # Create some real news data
        self.real_data = [{'title': 'New study shows benefits of exercise',
                           'text': 'A new study has found that regular exercise can have a number of benefits for '
                                   'your health.', 'label': 'real'},
                          {'title': 'Stock market hits all-time high',
                           'text': 'The stock market has reached an all-time high, with many investors seeing '
                                   'significant gains.','label': 'real'},
                          {'title': 'Record-breaking temperatures',
                           'text': 'Temperatures across the country have reached record-breaking levels this week.',
                           'label': 'real'}]
        # Combine the real and fake news data
        self.data = self.fake_data + self.real_data
        random.shuffle(self.data)

    def test_preprocess(self):
        text = "This is a test string. It has punctuation marks and capital letters!"
        expected_output = ['test', 'string', 'punctuation', 'marks', 'cap']
        self.assertEqual(preprocess(text), expected_output)

    def test_preprocess_text(self):
        article = {'title': 'New Study Finds Exercise Reduces Stress',
                   'text': 'A new study has found that regular exercise can significantly reduce stress levels.'}
        expected_output = {'title': 'New Study Finds Exercise Reduces Stress',
                           'text': ['studi', 'regular', 'exercis', 'reduc']}
        self.assertEqual(preprocess_text(article), expected_output)

    def test_train_test_split(self):
        X_train, y_train, X_test, y_test = train_test_split(self.data, split_factor=0.8)
        # Check that the total number of samples in X_train and X_test is equal to the total number of samples in the
        # original data
        self.assertEqual(len(X_train) + len(X_test), len(self.data))
        # Check that the ratio of training samples to testing samples is approximately equal to the split factor
        self.assertAlmostEqual(len(X_train) / (len(X_train) + len(X_test)), 0.6666666666666666, places=1)
