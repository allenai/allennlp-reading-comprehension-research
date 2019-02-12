from typing import Any, Dict, List, Optional
import torch
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.models.reading_comprehension.bidaf import BidirectionalAttentionFlow
from allennlp.modules import Highway
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TimeDistributed, TextFieldEmbedder
from allennlp.modules.matrix_attention.legacy_matrix_attention import LegacyMatrixAttention
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from reading_comprehension.drop_metrics import DropEmAndF1


@Model.register("bidaf_marginal")
class BiDAFMarginal(Model):
    """
    This class adapts the BiDAF model to do question answering on DROP dataset.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 num_highway_layers: int,
                 phrase_layer: Seq2SeqEncoder,
                 similarity_function: SimilarityFunction,
                 modeling_layer: Seq2SeqEncoder,
                 span_end_encoder: Seq2SeqEncoder,
                 dropout: float = 0.2,
                 mask_lstms: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(BiDAFMarginal, self).__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        self._highway_layer = TimeDistributed(Highway(text_field_embedder.get_output_dim(),
                                                      num_highway_layers))
        self._phrase_layer = phrase_layer
        self._matrix_attention = LegacyMatrixAttention(similarity_function)
        self._modeling_layer = modeling_layer
        self._span_end_encoder = span_end_encoder

        encoding_dim = phrase_layer.get_output_dim()
        modeling_dim = modeling_layer.get_output_dim()
        span_start_input_dim = encoding_dim * 4 + modeling_dim
        self._span_start_predictor = TimeDistributed(torch.nn.Linear(span_start_input_dim, 1))

        span_end_encoding_dim = span_end_encoder.get_output_dim()
        span_end_input_dim = encoding_dim * 4 + span_end_encoding_dim
        self._span_end_predictor = TimeDistributed(torch.nn.Linear(span_end_input_dim, 1))

        # Bidaf has lots of layer dimensions which need to match up - these aren't necessarily
        # obvious from the configuration files, so we check here.
        check_dimensions_match(modeling_layer.get_input_dim(), 4 * encoding_dim,
                               "modeling layer input dim", "4 * encoding dim")
        check_dimensions_match(text_field_embedder.get_output_dim(), phrase_layer.get_input_dim(),
                               "text field embedder output dim", "phrase layer input dim")
        check_dimensions_match(span_end_encoder.get_input_dim(), 4 * encoding_dim + 3 * modeling_dim,
                               "span end encoder input dim", "4 * encoding dim + 3 * modeling dim")

        self._drop_metrics = DropEmAndF1()
        self._dropout = torch.nn.Dropout(p=dropout)
        self._mask_lstms = mask_lstms

        initializer(self)

    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                numbers_in_passage: Dict[str, torch.LongTensor],
                number_indices: torch.LongTensor,
                answer_as_passage_spans: torch.LongTensor = None,
                answer_as_question_spans: torch.LongTensor = None,
                answer_as_add_sub_expressions: torch.LongTensor = None,
                answer_as_counts: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ, unused-argument

        embedded_question = self._highway_layer(self._text_field_embedder(question))
        embedded_passage = self._highway_layer(self._text_field_embedder(passage))
        batch_size = embedded_question.size(0)
        passage_length = embedded_passage.size(1)
        question_mask = util.get_text_field_mask(question).float()
        passage_mask = util.get_text_field_mask(passage).float()
        question_lstm_mask = question_mask if self._mask_lstms else None
        passage_lstm_mask = passage_mask if self._mask_lstms else None

        encoded_question = self._dropout(self._phrase_layer(embedded_question, question_lstm_mask))
        encoded_passage = self._dropout(self._phrase_layer(embedded_passage, passage_lstm_mask))
        encoding_dim = encoded_question.size(-1)

        # Shape: (batch_size, passage_length, question_length)
        passage_question_similarity = self._matrix_attention(encoded_passage, encoded_question)
        # Shape: (batch_size, passage_length, question_length)
        passage_question_attention = util.masked_softmax(passage_question_similarity, question_mask)
        # Shape: (batch_size, passage_length, encoding_dim)
        passage_question_vectors = util.weighted_sum(encoded_question, passage_question_attention)

        # We replace masked values with something really negative here, so they don't affect the
        # max below.
        masked_similarity = util.replace_masked_values(passage_question_similarity,
                                                       question_mask.unsqueeze(1),
                                                       -1e7)
        # Shape: (batch_size, passage_length)
        question_passage_similarity = masked_similarity.max(dim=-1)[0].squeeze(-1)
        # Shape: (batch_size, passage_length)
        question_passage_attention = util.masked_softmax(question_passage_similarity, passage_mask)
        # Shape: (batch_size, encoding_dim)
        question_passage_vector = util.weighted_sum(encoded_passage, question_passage_attention)
        # Shape: (batch_size, passage_length, encoding_dim)
        tiled_question_passage_vector = question_passage_vector.unsqueeze(1).expand(batch_size,
                                                                                    passage_length,
                                                                                    encoding_dim)

        # Shape: (batch_size, passage_length, encoding_dim * 4)
        final_merged_passage = torch.cat([encoded_passage,
                                          passage_question_vectors,
                                          encoded_passage * passage_question_vectors,
                                          encoded_passage * tiled_question_passage_vector],
                                         dim=-1)

        modeled_passage = self._dropout(self._modeling_layer(final_merged_passage, passage_lstm_mask))
        modeling_dim = modeled_passage.size(-1)

        # Shape: (batch_size, passage_length, encoding_dim * 4 + modeling_dim))
        span_start_input = self._dropout(torch.cat([final_merged_passage, modeled_passage], dim=-1))
        # Shape: (batch_size, passage_length)
        span_start_logits = self._span_start_predictor(span_start_input).squeeze(-1)
        # Shape: (batch_size, passage_length)
        span_start_probs = util.masked_softmax(span_start_logits, passage_mask)

        # Shape: (batch_size, modeling_dim)
        span_start_representation = util.weighted_sum(modeled_passage, span_start_probs)
        # Shape: (batch_size, passage_length, modeling_dim)
        tiled_start_representation = span_start_representation.unsqueeze(1).expand(batch_size,
                                                                                   passage_length,
                                                                                   modeling_dim)

        # Shape: (batch_size, passage_length, encoding_dim * 4 + modeling_dim * 3)
        span_end_representation = torch.cat([final_merged_passage,
                                             modeled_passage,
                                             tiled_start_representation,
                                             modeled_passage * tiled_start_representation],
                                            dim=-1)
        # Shape: (batch_size, passage_length, encoding_dim)
        encoded_span_end = self._dropout(self._span_end_encoder(span_end_representation,
                                                                passage_lstm_mask))
        # Shape: (batch_size, passage_length, encoding_dim * 4 + span_end_encoding_dim)
        span_end_input = self._dropout(torch.cat([final_merged_passage, encoded_span_end], dim=-1))
        span_end_logits = self._span_end_predictor(span_end_input).squeeze(-1)

        # Shape: (batch_size, passage_length)
        passage_span_start_log_probs = util.masked_log_softmax(span_start_logits, passage_mask)
        passage_span_end_log_probs = util.masked_log_softmax(span_end_logits, passage_mask)

        passage_span_start_logits = util.replace_masked_values(span_start_logits, passage_mask, -1e7)
        passage_span_end_logits = util.replace_masked_values(span_end_logits, passage_mask, -1e7)
        # Shape: (batch_size, 2)
        best_passage_span = \
            BidirectionalAttentionFlow.get_best_span(passage_span_start_logits, passage_span_end_logits)

        output_dict = {"passage_question_attention": passage_question_attention,
                       "passage_span_start_probs": passage_span_start_log_probs.exp(),
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
