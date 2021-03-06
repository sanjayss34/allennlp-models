import logging
from typing import Dict, List, Iterable, Tuple, Any
import random
from collections import defaultdict

from overrides import overrides
from transformers import AutoTokenizer, AutoConfig
# from transformers.tokenization_bert import BertTokenizer
import numpy as np

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer, PretrainedTransformerIndexer, PretrainedTransformerMismatchedIndexer
from allennlp.data.tokenizers import Token, PretrainedTransformerTokenizer
from allennlp_models.common.ontonotes import Ontonotes, OntonotesSentence

from allennlp_models.coref.util import make_coref_instance

logger = logging.getLogger(__name__)


def _convert_tags_to_wordpiece_tags(tags: List[str], offsets: List[int]) -> List[str]:
    """
    Converts a series of BIO tags to account for a wordpiece tokenizer,
    extending/modifying BIO tags where appropriate to deal with words which
    are split into multiple wordpieces by the tokenizer.

    This is only used if you pass a `bert_model_name` to the dataset reader below.

    # Parameters

    tags : `List[str]`
        The BIO formatted tags to convert to BIO tags for wordpieces
    offsets : `List[int]`
        The wordpiece offsets.

    # Returns

    The new BIO tags.
    """
    new_tags = []
    j = 0
    for i, offset in enumerate(offsets):
        tag = tags[i]
        is_o = tag == "O"
        is_start = True
        while j < offset:
            if is_o:
                new_tags.append("O")

            elif tag.startswith("I"):
                new_tags.append(tag)

            elif is_start and tag.startswith("B"):
                new_tags.append(tag)
                is_start = False

            elif tag.startswith("B"):
                _, label = tag.split("-", 1)
                new_tags.append("I-" + label)
            j += 1

    # Add O tags for cls and sep tokens.
    return ["O"] + new_tags + ["O"]


def _convert_verb_indices_to_wordpiece_indices(verb_indices: List[int], offsets: List[int]):
    """
    Converts binary verb indicators to account for a wordpiece tokenizer,
    extending/modifying BIO tags where appropriate to deal with words which
    are split into multiple wordpieces by the tokenizer.

    This is only used if you pass a `bert_model_name` to the dataset reader below.

    # Parameters

    verb_indices : `List[int]`
        The binary verb indicators, 0 for not a verb, 1 for verb.
    offsets : `List[int]`
        The wordpiece offsets.

    # Returns

    The new verb indices.
    """
    j = 0
    new_verb_indices = []
    for i, offset in enumerate(offsets):
        indicator = verb_indices[i]
        while j < offset:
            new_verb_indices.append(indicator)
            j += 1

    # Add 0 indicators for cls and sep tokens.
    return [0] + new_verb_indices + [0]


@DatasetReader.register("srl")
class SrlReader(DatasetReader):
    """
    This DatasetReader is designed to read in the English OntoNotes v5.0 data
    for semantic role labelling. It returns a dataset of instances with the
    following fields:

    tokens : `TextField`
        The tokens in the sentence.
    verb_indicator : `SequenceLabelField`
        A sequence of binary indicators for whether the word is the verb for this frame.
    tags : `SequenceLabelField`
        A sequence of Propbank tags for the given verb in a BIO format.

    # Parameters

    token_indexers : `Dict[str, TokenIndexer]`, optional
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
        Default is `{"tokens": SingleIdTokenIndexer()}`.
    domain_identifier : `str`, (default = `None`)
        A string denoting a sub-domain of the Ontonotes 5.0 dataset to use. If present, only
        conll files under paths containing this domain identifier will be processed.
    bert_model_name : `Optional[str]`, (default = `None`)
        The BERT model to be wrapped. If you specify a bert_model here, then we will
        assume you want to use BERT throughout; we will use the bert tokenizer,
        and will expand your tags and verb indicators accordingly. If not,
        the tokens will be indexed as normal with the token_indexers.

    # Returns

    A `Dataset` of `Instances` for Semantic Role Labelling.
    """

    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        domain_identifier: str = None,
        bert_model_name: str = None,
        with_spans: bool = False,
        mismatched_tokens: bool = False,
        random_sample: bool = False,
        random_seed: int = None,
        limit: int = -1,
        print_violations: bool = False,
        label_namespace: str = "labels",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._token_indexers = token_indexers or {"tokens": PretrainedTransformerIndexer(model_name=bert_model_name)}
        if mismatched_tokens:
            self._token_indexers = {"tokens": PretrainedTransformerMismatchedIndexer(model_name=bert_model_name)}
            self._dummy_indexer = {"dummy_tokens": SingleIdTokenIndexer()}
        self._domain_identifier = domain_identifier
        self._with_spans = with_spans
        self._mismatched_tokens = mismatched_tokens
        self._limit = limit
        self._random_sample = random_sample
        self._random_seed = random_seed
        self._max_sequence_length = 0
        self._print_violations = print_violations
        self._label_namespace = label_namespace

        if bert_model_name is not None:
            self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
            if with_spans:
                self.bert_tokenizer_allennlp = PretrainedTransformerTokenizer(model_name=bert_model_name)
            self.lowercase_input = "uncased" in bert_model_name
            self.bert_config = AutoConfig.from_pretrained(bert_model_name)
        else:
            self.bert_tokenizer = None
            self.lowercase_input = False

    def _wordpiece_tokenize_input(
        self, tokens: List[str]
    ) -> Tuple[List[str], List[int], List[int]]:
        """
        Convert a list of tokens to wordpiece tokens and offsets, as well as adding
        BERT CLS and SEP tokens to the beginning and end of the sentence.

        A slight oddity with this function is that it also returns the wordpiece offsets
        corresponding to the _start_ of words as well as the end.

        We need both of these offsets (or at least, it's easiest to use both), because we need
        to convert the labels to tags using the end_offsets. However, when we are decoding a
        BIO sequence inside the SRL model itself, it's important that we use the start_offsets,
        because otherwise we might select an ill-formed BIO sequence from the BIO sequence on top of
        wordpieces (this happens in the case that a word is split into multiple word pieces,
        and then we take the last tag of the word, which might correspond to, e.g, I-V, which
        would not be allowed as it is not preceded by a B tag).

        For example:

        `annotate` will be bert tokenized as ["anno", "##tate"].
        If this is tagged as [B-V, I-V] as it should be, we need to select the
        _first_ wordpiece label to be the label for the token, because otherwise
        we may end up with invalid tag sequences (we cannot start a new tag with an I).

        # Returns

        wordpieces : `List[str]`
            The BERT wordpieces from the words in the sentence.
        end_offsets : `List[int]`
            Indices into wordpieces such that `[wordpieces[i] for i in end_offsets]`
            results in the end wordpiece of each word being chosen.
        start_offsets : `List[int]`
            Indices into wordpieces such that `[wordpieces[i] for i in start_offsets]`
            results in the start wordpiece of each word being chosen.
        """
        word_piece_tokens: List[str] = []
        end_offsets = []
        start_offsets = []
        cumulative = 0
        for token in tokens:
            if self.lowercase_input:
                token = token.lower()
            word_piece_ids = self.bert_tokenizer.encode_plus(token, add_special_tokens=False, return_tensors=None, return_offsets_mapping=False, return_attention_mask=False, return_token_type_ids=False)
            word_pieces = self.bert_tokenizer.convert_ids_to_tokens(word_piece_ids['input_ids'])
            start_offsets.append(cumulative + 1)
            cumulative += len(word_pieces)
            end_offsets.append(cumulative)
            word_piece_tokens.extend(word_pieces)

        wordpieces = [self.bert_tokenizer.cls_token]+word_piece_tokens+[self.bert_tokenizer.sep_token]

        return wordpieces, end_offsets, start_offsets

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        ontonotes_reader = Ontonotes()
        logger.info("Reading SRL instances from dataset files at: %s", file_path)
        if self._domain_identifier is not None:
            logger.info(
                "Filtering to only include file paths containing the %s domain",
                self._domain_identifier,
            )

        count = 0
        instances = []
        for index, sentence in enumerate(self._ontonotes_subset(
            ontonotes_reader, file_path, self._domain_identifier
        )):
            if self._limit > 0 and count >= self._limit and not self._random_sample:
                break
            tokens = [Token(t) for t in sentence.words]
            if not sentence.srl_frames:
                # Sentence contains no predicates.
                tags = ["O" for _ in tokens]
                verb_label = [0 for _ in tokens]
                count += 1
                instance = self.text_to_instance(tokens, verb_label, tags)
                if self._random_sample and self._limit > 0:
                    instances.append(instance)
                else:
                    yield instance
            else:
                for (_, tags) in sentence.srl_frames:
                    if self._limit > 0 and count >= self._limit and not self._random_sample:
                        break
                    count += 1
                    verb_indicator = [1 if label[-2:] == "-V" else 0 for label in tags]
                    if self._print_violations:
                        violation = False
                        counts = defaultdict(int)
                        for t in tags:
                            counts[t] += 1
                        for key in list(counts.keys()):
                            if key[:4] in {"B-R-", "B-C-"}:
                                if counts["B-"+key[4:]] == 0:
                                    violation = True
                                    break
                        if violation:
                            logger.info(tokens)
                            logger.info(tags)
                    instance = self.text_to_instance(tokens, verb_indicator, tags)
                    if self._random_sample and self._limit > 0:
                        instances.append(instance)
                    else:
                        yield instance
        if self._random_sample and self._limit > 0:
            random.seed(self._random_seed)
            sample = random.sample(instances, self._limit)
            for instance in sample:
                yield instance

    @staticmethod
    def _ontonotes_subset(
        ontonotes_reader: Ontonotes, file_path: str, domain_identifier: str
    ) -> Iterable[OntonotesSentence]:
        """
        Iterates over the Ontonotes 5.0 dataset using an optional domain identifier.
        If the domain identifier is present, only examples which contain the domain
        identifier in the file path are yielded.
        """
        for conll_file in ontonotes_reader.dataset_path_iterator(file_path):
            if domain_identifier is None or f"/{domain_identifier}/" in conll_file:
                yield from ontonotes_reader.sentence_iterator(conll_file)

    def text_to_instance(  # type: ignore
        self, tokens: List[Token], verb_label: List[int], tags: List[str] = None
    ) -> Instance:
        """
        We take `pre-tokenized` input here, along with a verb label.  The verb label should be a
        one-hot binary vector, the same length as the tokens, indicating the position of the verb
        to find arguments for.
        """
        metadata_dict: Dict[str, Any] = {}
        if self.bert_tokenizer is not None:
            wordpieces, offsets, start_offsets = self._wordpiece_tokenize_input(
                [t.text for t in tokens]
            )
            self._max_sequence_length = max(self._max_sequence_length, len(wordpieces))
            # print(self._max_sequence_length)
            new_verbs = _convert_verb_indices_to_wordpiece_indices(verb_label, offsets)
            if not self._with_spans and not self._mismatched_tokens:
                verb_tokens = [token for token, v in zip(wordpieces, new_verbs) if v == 1]
                wordpieces += verb_tokens+[self.bert_tokenizer.sep_token]
                new_verbs += [0 for _ in range(len(verb_tokens)+1)]
            # In order to override the indexing mechanism, we need to set the `text_id`
            # attribute directly. This causes the indexing to use this id.
            vocab = self.bert_tokenizer.get_vocab()
            text_field = TextField(
                [Token(t, text_id=vocab[t]) for t in wordpieces],
                token_indexers={"tokens": self._token_indexers["tokens"]._matched_indexer}
            )
            if self._mismatched_tokens:
                sep_index = len(tokens)+1
                real_text_field = TextField(tokens, token_indexers=self._token_indexers)
                if self.bert_config.type_vocab_size == 1:
                    text_field = real_text_field
                    new_verbs = verb_label
            else:
                sep_index = wordpieces.index(self.bert_tokenizer.sep_token)
                real_text_field = text_field
            metadata_dict["offsets"] = start_offsets
            verb_indicator = SequenceLabelField(new_verbs, text_field, label_namespace=self._label_namespace)

        else:
            text_field = TextField(tokens, token_indexers=self._token_indexers)
            verb_indicator = SequenceLabelField(verb_label, text_field)

        fields: Dict[str, Field] = {}
        fields["tokens"] = real_text_field
        fields["verb_indicator"] = verb_indicator
        fields["sentence_end"] = ArrayField(np.array(sep_index+1, dtype=np.int64), dtype=np.int64)

        if all(x == 0 for x in verb_label):
            verb = None
            verb_index = None
        else:
            verb_index = verb_label.index(1)
            verb = tokens[verb_index].text

        metadata_dict["words"] = [x.text for x in tokens]
        metadata_dict["verb"] = verb
        metadata_dict["verb_index"] = verb_index

        if self._with_spans:
            if tags:
                new_tags = _convert_tags_to_wordpiece_tags(tags, offsets)
                tag_index_dict = {}
                cur_span_start = None
                for i in range(len(new_tags)):
                    if new_tags[i] != 'O':
                        if new_tags[i][0] == 'B':
                            if cur_span_start is not None:
                                tag_index_dict[(cur_span_start, i-1)] = new_tags[i-1][2:]
                            cur_span_start = i
                    else:
                        if cur_span_start is not None:
                            tag_index_dict[(cur_span_start, i-1)] = new_tags[i-1][2:]
                        cur_span_start = None
            # print(tag_index_dict)
            coref_instance = make_coref_instance([wordpieces], self._token_indexers, 30, span_label_map=tag_index_dict)
            fields["tokens"] = coref_instance.fields["text"]
            text_field = fields["tokens"]
            fields["verb_indicator"] = SequenceLabelField(new_verbs, fields["tokens"])
            fields["spans"] = coref_instance.fields["spans"]
            fields["span_labels"] = coref_instance.fields["span_labels"]
            # fields["span_mask"] = ArrayField(np.array([1 for _ in range(len(fields['spans'].field_list))], dtype=np.int64), dtype=np.int64)
            # print([(span.span_start, span.span_end, label) for span, label in zip(fields['spans'].field_list, fields['span_labels'].labels) if label != "O"])

        if tags:
            if self.bert_tokenizer is not None:
                if not self._mismatched_tokens:
                    new_tags = _convert_tags_to_wordpiece_tags(tags, offsets)
                    new_tags += ["O" for _ in range(len(wordpieces)-len(new_tags))]
                else:
                    new_tags = tags
                fields["tags"] = SequenceLabelField(new_tags, real_text_field, label_namespace=self._label_namespace)
            else:
                fields["tags"] = SequenceLabelField(tags, text_field)
            metadata_dict["gold_tags"] = tags

        fields["metadata"] = MetadataField(metadata_dict)
        return Instance(fields)
