import string
import re
from typing import Tuple, List, Union
from overrides import overrides
from allennlp.tools.squad_eval import metric_max_over_ground_truths
from allennlp.training.metrics.metric import Metric
from reading_comprehension.data.drop_official_evaluate import get_metrics as drop_em_and_f1
from reading_comprehension.data.drop_official_evaluate import to_string as convert_annotation_to_string


STOPWORDS = set(["a", "an", "the"])
PUNCTUATIONS = set(string.punctuation)


def string_to_bag(raw_text):
    text = raw_text.lower()
    text_tokens = set()
    for token in text.strip().split(" "):
        if not re.match(r"\d*\.\d+", token):
            token = ''.join(ch for ch in token if ch not in PUNCTUATIONS)
        if token != '':
            text_tokens.add(token)
    return set(text_tokens) - STOPWORDS


def bag_of_words_exact_match(prediction: str, ground_truth: str):
    return string_to_bag(prediction) == string_to_bag(ground_truth)


def bag_of_words_f1(prediction: str, ground_truth: str):
    prediction_bag = string_to_bag(prediction)
    gold_bag = string_to_bag(ground_truth)
    hit = len(gold_bag.intersection(prediction_bag))
    if hit > 0:
        precision = 1.0 * hit / len(prediction_bag)
        recall = 1.0 * hit / len(gold_bag)
        return 2.0 * precision * recall / (precision + recall)
    else:
        return 0.0


@Metric.register("drop")
class DropEmAndF1(Metric):
    """
    This :class:`Metric` takes the best span string computed by a model, along with the answer
    strings labeled in the data, and computed exact match and F1 score based on bag of words
    """
    def __init__(self) -> None:
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._count = 0

    @overrides
    def __call__(self, prediction: Union[str, List], ground_truths: List):
        """
        Parameters
        ----------
        prediction: ``Union[str, List]``
            The predicted answer from the model evaluated. This could be a string, or a list of string
            when multiple spans are predicted as answer.
        ground_truths: ``List``
            All the ground truth answer annotations.
        """
        ground_truth_answer_strings = [convert_annotation_to_string(annotation)[0] for annotation in ground_truths]
        # pylint: disable=unused-variable
        ground_truth_answer_types = [convert_annotation_to_string(annotation)[1] for annotation in ground_truths]
        exact_match, f1_score = metric_max_over_ground_truths(
                drop_em_and_f1,
                prediction,
                ground_truth_answer_strings
        )
        self._total_em += exact_match
        self._total_f1 += f1_score
        self._count += 1

    @overrides
    def get_metric(self, reset: bool = False) -> Tuple[float, float]:
        """
        Returns
        -------
        Average exact match and F1 score (in that order) as computed by the official SQuAD script
        over all inputs.
        """
        exact_match = self._total_em / self._count if self._count > 0 else 0
        f1_score = self._total_f1 / self._count if self._count > 0 else 0
        if reset:
            self.reset()
        return exact_match, f1_score

    @overrides
    def reset(self):
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._count = 0

    def __str__(self):
        return f"DropEmAndF1(em={self._total_em}, f1={self._total_f1})"
