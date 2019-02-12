from typing import Any, Dict, List, Optional
import logging
import torch
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.models.reading_comprehension.util import get_best_span
from reading_comprehension.drop_metrics import DropEmAndF1

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("bert_rc_marginal")
class BertRcMarginal(Model):
    """
    This class adapts the BERT RC model to do question answering on DROP dataset.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 dropout: float = 0.0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        self._span_start_predictor = torch.nn.Linear(self._text_field_embedder.get_output_dim(), 1)
        self._span_end_predictor = torch.nn.Linear(self._text_field_embedder.get_output_dim(), 1)

        self._drop_metrics = DropEmAndF1()
        self._dropout = torch.nn.Dropout(p=dropout)

        initializer(self)

    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                question_and_passage: Dict[str, torch.LongTensor],
                answer_as_passage_spans: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ, unused-argument

        # logger.info("="*10)
        # logger.info([len(metadata[i]["passage_tokens"]) for i in range(len(metadata))])
        # logger.info([len(metadata[i]["question_tokens"]) for i in range(len(metadata))])
        # logger.info(question_and_passage["bert"].shape)

        # The segment labels should be as following:
        # <CLS> + question_word_pieces + <SEP> + passage_word_pieces + <SEP>
        # 0                0               0              1              1
        # We get this in a tricky way here
        expanded_question_bert_tensor = torch.zeros_like(question_and_passage["bert"])
        expanded_question_bert_tensor[:, :question["bert"].shape[1]] = question["bert"]
        segment_labels = (question_and_passage["bert"] - expanded_question_bert_tensor > 0).long()
        question_and_passage["segment_labels"] = segment_labels
        embedded_question_and_passage = self._text_field_embedder(question_and_passage)

        # We also get the passage mask for the concatenated question and passage in a similar way
        expanded_question_mask = torch.zeros_like(question_and_passage["mask"])
        # We shift the 1s to one column right here, to mask the [SEP] token in the middle
        expanded_question_mask[:, 1:question["mask"].shape[1]+1] = question["mask"]
        expanded_question_mask[:, 0] = 1
        passage_mask = question_and_passage["mask"] - expanded_question_mask

        batch_size = embedded_question_and_passage.size(0)

        span_start_logits = self._span_start_predictor(embedded_question_and_passage).squeeze(-1)
        span_end_logits = self._span_end_predictor(embedded_question_and_passage).squeeze(-1)

        # Shape: (batch_size, passage_length)
        passage_span_start_log_probs = util.masked_log_softmax(span_start_logits, passage_mask)
        passage_span_end_log_probs = util.masked_log_softmax(span_end_logits, passage_mask)

        passage_span_start_logits = util.replace_masked_values(span_start_logits, passage_mask, -1e32)
        passage_span_end_logits = util.replace_masked_values(span_end_logits, passage_mask, -1e32)
        best_passage_span = get_best_span(passage_span_start_logits, passage_span_end_logits)

        output_dict = {"passage_span_start_probs": passage_span_start_log_probs.exp(),
                       "passage_span_end_probs": passage_span_end_log_probs.exp()}

        # If answer is given, compute the loss for training.
        if answer_as_passage_spans is not None:
            # Shape: (batch_size, # of answer spans)
            gold_passage_span_starts = answer_as_passage_spans[:, :, 0]
            gold_passage_span_ends = answer_as_passage_spans[:, :, 1]
            # Some spans are padded with index -1,
            # so we clamp those paddings to 0 and then mask after `torch.gather()`.
            gold_passage_span_mask = (gold_passage_span_starts != -1).long()
            clamped_gold_passage_span_starts = util.replace_masked_values(gold_passage_span_starts,
                                                                          gold_passage_span_mask,
                                                                          0)
            clamped_gold_passage_span_ends = util.replace_masked_values(gold_passage_span_ends,
                                                                        gold_passage_span_mask,
                                                                        0)
            # Shape: (batch_size, # of answer spans)
            log_likelihood_for_passage_span_starts = \
                torch.gather(passage_span_start_log_probs, 1, clamped_gold_passage_span_starts)
            log_likelihood_for_passage_span_ends = \
                torch.gather(passage_span_end_log_probs, 1, clamped_gold_passage_span_ends)
            # Shape: (batch_size, # of answer spans)
            log_likelihood_for_passage_spans = \
                log_likelihood_for_passage_span_starts + log_likelihood_for_passage_span_ends
            # For those padded spans, we set their log probabilities to be very small negative value
            log_likelihood_for_passage_spans = \
                util.replace_masked_values(log_likelihood_for_passage_spans, gold_passage_span_mask, -1e32)
            # Shape: (batch_size, )
            log_marginal_likelihood_for_passage_span = util.logsumexp(log_likelihood_for_passage_spans)
            output_dict["loss"] = - log_marginal_likelihood_for_passage_span.mean()

        # Compute the metrics and add the tokenized input to the output.
        if metadata is not None:
            output_dict["question_id"] = []
            output_dict["answer"] = []
            question_tokens = []
            passage_tokens = []
            for i in range(batch_size):
                question_tokens.append(metadata[i]['question_tokens'])
                passage_tokens.append(metadata[i]['passage_tokens'])
                # We did not consider multi-mention answers here
                passage_str = metadata[i]['original_passage']
                offsets = metadata[i]['passage_token_offsets']
                predicted_span = tuple(best_passage_span[i].detach().cpu().numpy())
                # Remove the offsets of question tokens and the [SEP] token
                predicted_span = (predicted_span[0] - len(metadata[i]['question_tokens']) - 1,
                                  predicted_span[1] - len(metadata[i]['question_tokens']) - 1)
                start_offset = offsets[predicted_span[0]][0]
                end_offset = offsets[predicted_span[1]][1]
                best_answer_str = passage_str[start_offset:end_offset]
                output_dict["question_id"].append(metadata[i]["question_id"])
                output_dict["answer"].append(best_answer_str)
                answer_annotations = metadata[i].get('answer_annotations', [])
                if answer_annotations:
                    self._drop_metrics(best_answer_str, answer_annotations)
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        exact_match, f1_score = self._drop_metrics.get_metric(reset)
        return {'em': exact_match, 'f1': f1_score}
