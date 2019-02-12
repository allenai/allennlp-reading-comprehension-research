from typing import Any, Dict, List, Optional
import torch
from torch.nn.functional import nll_loss
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Highway
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import BooleanAccuracy, CategoricalAccuracy
from allennlp.models.reading_comprehension.util import get_best_span
from reading_comprehension.drop_metrics import DropEmAndF1


# TODO: Change this to marginal loss
@Model.register("passage_only")
class PassageOnlyRcModel(Model):
    """
    This class will encode the passage using a encoder, and then predict a span for
    answer without considering the question.

    If you want to test the question-only baseline, just replace the passage with question
    when loading data in the data reader.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 num_highway_layers: int,
                 encoding_layer: Seq2SeqEncoder,
                 dropout_prob: float = 0.1,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        text_embed_dim = text_field_embedder.get_output_dim()
        encoding_in_dim = encoding_layer.get_input_dim()
        encoding_out_dim = encoding_layer.get_output_dim()

        self._text_field_embedder = text_field_embedder
        self._embedding_proj_layer = torch.nn.Linear(text_embed_dim, encoding_in_dim)
        self._highway_layer = Highway(encoding_in_dim, num_highway_layers)

        self._encoding_layer = encoding_layer

        self._span_start_predictor = torch.nn.Linear(encoding_out_dim * 2, 1)
        self._span_end_predictor = torch.nn.Linear(encoding_out_dim * 2, 1)

        self._span_start_accuracy = CategoricalAccuracy()
        self._span_end_accuracy = CategoricalAccuracy()
        self._span_accuracy = BooleanAccuracy()
        self._drop_metrics = DropEmAndF1()
        self._dropout = torch.nn.Dropout(p=dropout_prob)

        initializer(self)

    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                span_start: torch.IntTensor = None,
                span_end: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ, unused-argument
        passage_mask = util.get_text_field_mask(passage).float()
        embedded_passage = self._dropout(self._text_field_embedder(passage))
        batch_size = embedded_passage.size(0)
        embedded_passage = self._highway_layer(self._embedding_proj_layer(embedded_passage))

        encoded_passage_list = [embedded_passage]
        for _ in range(3):
            encoded_passage = self._dropout(self._encoding_layer(encoded_passage_list[-1], passage_mask))
            encoded_passage_list.append(encoded_passage)

        # Shape: (batch_size, passage_length, modeling_dim * 2))
        span_start_input = torch.cat([encoded_passage_list[-3], encoded_passage_list[-2]], dim=-1)
        # Shape: (batch_size, passage_length)
        span_start_logits = self._span_start_predictor(span_start_input).squeeze(-1)
        # Shape: (batch_size, passage_length, modeling_dim * 2)
        span_end_input = torch.cat([encoded_passage_list[-3], encoded_passage_list[-1]], dim=-1)
        span_end_logits = self._span_end_predictor(span_end_input).squeeze(-1)

        span_start_logits = util.replace_masked_values(span_start_logits, passage_mask, -1e32)
        span_end_logits = util.replace_masked_values(span_end_logits, passage_mask, -1e32)
        best_span = get_best_span(span_start_logits, span_end_logits)

        output_dict = {}

        # Compute the loss for training.
        if span_start is not None:
            loss = nll_loss(util.masked_log_softmax(span_start_logits, passage_mask), span_start.squeeze(-1))
            self._span_start_accuracy(span_start_logits, span_start.squeeze(-1))
            loss += nll_loss(util.masked_log_softmax(span_end_logits, passage_mask), span_end.squeeze(-1))
            self._span_end_accuracy(span_end_logits, span_end.squeeze(-1))
            self._span_accuracy(best_span, torch.stack([span_start, span_end], -1))
            output_dict["loss"] = loss

        # Compute the metrics and add the tokenized input to the output.
        if metadata is not None:
            output_dict["question_id"] = []
            output_dict["answer"] = []
            question_tokens = []
            passage_tokens = []
            for i in range(batch_size):
                question_tokens.append(metadata[i]['question_tokens'])
                passage_tokens.append(metadata[i]['passage_tokens'])
                passage_str = metadata[i]['original_passage']
                offsets = metadata[i]['token_offsets']
                predicted_span = tuple(best_span[i].detach().cpu().numpy())
                start_offset = offsets[predicted_span[0]][0]
                end_offset = offsets[predicted_span[1]][1]
                best_span_string = passage_str[start_offset:end_offset]
                output_dict["question_id"].append(metadata[i]["question_id"])
                output_dict["answer"].append(best_span_string)
                answer_annotations = metadata[i].get('answer_annotations', [])
                if answer_annotations:
                    self._drop_metrics(best_span_string, answer_annotations)
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        exact_match, f1_score = self._drop_metrics.get_metric(reset)
        return {
                'start_acc': self._span_start_accuracy.get_metric(reset),
                'end_acc': self._span_end_accuracy.get_metric(reset),
                'span_acc': self._span_accuracy.get_metric(reset),
                'em': exact_match,
                'f1': f1_score,
                }
