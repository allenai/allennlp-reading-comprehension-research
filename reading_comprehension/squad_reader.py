import json
import logging
from typing import Dict, List, Tuple, Optional, Iterable

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.reading_comprehension import util
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("squad_limited")
class SquadReader(DatasetReader):
    """
    This class is very similar to the original
    `allennlp.data.dataset_readers.reading_comprehension.squad.SquadReader` in AllenNLP excpet that:
    1. We supports limiting the maximum passage length and question length for training and evaluation.
    2. During training, we drop those examples with all answer spans exceeding the maximum passage
       length (invalid examples); During testing, we use (0, 0) as the gold answer span for these
       invalid examples.

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the question and the passage.  See :class:`Tokenizer`.
        Default is ```WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        We similarly use this for both the question and the passage.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    lazy : ``bool``, optional (default=False)
        If this is true, ``instances()`` will return an object whose ``__iter__`` method
        reloads the dataset each time it's called. Otherwise, ``instances()`` returns a list.
    passage_length_limit : ``int``, optional (default=None)
        if specified, we will cut the passage if the length of passage exceeds this limit.
    question_length_limit : ``int``, optional (default=None)
        if specified, we will cut the question if the length of passage exceeds this limit.
    passage_length_limit_for_evaluation : ``int``, optional (default=None)
        if specified, we will use this limit instead of the ``passage_length_limit`` during evaluation.
    question_length_limit_for_evaluation : ``int``, optional (default=None)
        if specified, we will use this limit instead of the ``question_length_limit`` during evaluation.
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 passage_length_limit: int = None,
                 question_length_limit: int = None,
                 passage_length_limit_for_evaluation: int = None,
                 question_length_limit_for_evaluation: int = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.passage_length_limit = passage_length_limit
        self.question_length_limit = question_length_limit
        self.passage_length_limit_for_eval = passage_length_limit_for_evaluation or passage_length_limit
        self.question_length_limit_for_eval = question_length_limit_for_evaluation or question_length_limit

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        """
        Similar to the `_read` function in AllenNLP's SquadReader except that
        we first determine whether the file read is for training or evaluation,
        then we will cut the passage according to different length limits and decide
        whether to keep the invalid examples or not.
        """
        # if `file_path` is a URL, redirect to the cache
        is_train = 'train' in str(file_path)
        file_path = cached_path(file_path)

        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset_json = json.load(dataset_file)
            dataset = dataset_json['data']
        logger.info("Reading the dataset")
        for article in dataset:
            for paragraph_json in article['paragraphs']:
                paragraph = paragraph_json["context"]
                tokenized_paragraph = self._tokenizer.tokenize(paragraph)

                for question_answer in paragraph_json['qas']:
                    question_text = question_answer["question"].strip().replace("\n", "")
                    answer_texts = [answer['text'] for answer in question_answer['answers']]
                    span_starts = [answer['answer_start'] for answer in question_answer['answers']]
                    span_ends = [start + len(answer) for start, answer in zip(span_starts, answer_texts)]
                    if is_train:
                        instance = self.text_to_instance(question_text,
                                                         paragraph,
                                                         zip(span_starts, span_ends),
                                                         answer_texts,
                                                         tokenized_paragraph,
                                                         max_passage_len=self.passage_length_limit,
                                                         max_question_len=self.question_length_limit,
                                                         drop_invalid=True)
                    else:
                        instance = self.text_to_instance(question_text,
                                                         paragraph,
                                                         zip(span_starts, span_ends),
                                                         answer_texts,
                                                         tokenized_paragraph,
                                                         max_passage_len=self.passage_length_limit_for_eval,
                                                         max_question_len=self.question_length_limit_for_eval,
                                                         drop_invalid=False)
                    if instance is not None:
                        yield instance

    @overrides
    def text_to_instance(self,  # type: ignore
                         question_text: str,
                         passage_text: str,
                         char_spans: List[Tuple[int, int]] = None,
                         answer_texts: List[str] = None,
                         passage_tokens: List[Token] = None,
                         max_passage_len: int = None,
                         max_question_len: int = None,
                         drop_invalid: bool = False) -> Optional[Instance]:
        """
        We cut the passage and question according to `max_passage_len` and `max_question_len` here.
        We will drop the invalid examples if `drop_invalid` equals to true.
        """
        # pylint: disable=arguments-differ
        if not passage_tokens:
            passage_tokens = self._tokenizer.tokenize(passage_text)
        question_tokens = self._tokenizer.tokenize(question_text)
        if max_passage_len is not None:
            passage_tokens = passage_tokens[: max_passage_len]
        if max_question_len is not None:
            question_tokens = question_tokens[: max_question_len]
        char_spans = char_spans or []
        # We need to convert character indices in `passage_text` to token indices in
        # `passage_tokens`, as the latter is what we'll actually use for supervision.
        token_spans: List[Tuple[int, int]] = []
        passage_offsets = [(token.idx, token.idx + len(token.text)) for token in passage_tokens]
        for char_span_start, char_span_end in char_spans:
            if char_span_end > passage_offsets[-1][1]:
                continue
            (span_start, span_end), error = util.char_span_to_token_span(passage_offsets,
                                                                         (char_span_start, char_span_end))
            if error:
                logger.debug("Passage: %s", passage_text)
                logger.debug("Passage tokens: %s", passage_tokens)
                logger.debug("Question text: %s", question_text)
                logger.debug("Answer span: (%d, %d)", char_span_start, char_span_end)
                logger.debug("Token span: (%d, %d)", span_start, span_end)
                logger.debug("Tokens in answer: %s", passage_tokens[span_start:span_end + 1])
                logger.debug("Answer: %s", passage_text[char_span_start:char_span_end])
            token_spans.append((span_start, span_end))
        if not token_spans:
            if drop_invalid:
                return None
            else:
                token_spans.append((0, 0))
        return util.make_reading_comprehension_instance(question_tokens,
                                                        passage_tokens,
                                                        self._token_indexers,
                                                        passage_text,
                                                        token_spans,
                                                        answer_texts)
