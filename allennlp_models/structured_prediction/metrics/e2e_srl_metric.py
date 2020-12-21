from typing import List, Tuple

from allennlp.training.metrics import Metric

class E2eSrlMetric(Metric):
    def __init__(self):
        self._true_positives = 0
        self._false_positives = 0
        self._false_negatives = 0

    def __call__(self,
                 predictions: List[List[Tuple[int, Tuple[int, int, str]]]],
                 gold: List[List[Tuple[int, Tuple[int, int, str]]]]) -> None:
        for batch_index in range(len(predictions)):
            prediction_set = set(predictions[batch_index])
            gold_set = set(gold[batch_index])
            intersection = prediction_set.intersection(gold_set)
            self._true_positives += len(intersection)
            self._false_positives += len(prediction_set-intersection)
            self._false_negatives += len(gold_set-intersection)

    def get_metric(self, reset: bool = False):
        if self._true_positives == 0:
            return 0.0
        recall = self._true_positives/(self._true_positives+self._false_negatives)
        precision = self._true_positives/(self._true_positives+self._false_positives)
        f1 = 2*precision*recall/(precision+recall)
        return f1
