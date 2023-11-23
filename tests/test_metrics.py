import unittest
from metrics import *


class TestMetrics(unittest.TestCase):

    def test_confusion_matrix_and_accuracy(self):
        true_labels = ['real', 'fake', 'real', 'fake', 'real', 'fake', 'real', 'fake', 'real', 'fake']
        predictions = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1]

        cm, acc = confusion_matrix_and_accuracy(true_labels, predictions)
        self.assertEqual(cm[0][0], 5)
        self.assertEqual(cm[0][1], 0)
        self.assertEqual(cm[1][0], 0)
        self.assertEqual(cm[1][1], 5)
        self.assertEqual(acc, 1.0)

    def test_calculate_accuracy_from_confusion_matrix(self):
        tp = 20
        fp = 10
        fn = 5
        tn = 65

        acc = calculate_accuracy_from_confusion_matrix(tp, fp, fn, tn)
        self.assertEqual(acc, 0.85)

    def test_calculate_precision(self):
        tp = 20
        fp = 10

        precision = calculate_precision(tp, fp)
        self.assertEqual(precision, 0.6666666666666666)

    def test_calculate_recall(self):
        tp = 20
        fn = 5

        recall = calculate_recall(tp, fn)
        self.assertEqual(recall, 0.8)

    def test_calculate_f1_score(self):
        precision = 0.75
        recall = 0.6

        f1_score = calculate_f1_score(precision, recall)
        self.assertEqual(f1_score, 0.6666666666666665)


if __name__ == '__main__':
    unittest.main()
