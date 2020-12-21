import logging
import collections
from typing import Dict, List, Optional, Tuple, DefaultDict
import os

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer

from allennlp_models.common.ontonotes import Ontonotes
from allennlp_models.coref.util import make_coref_instance

logger = logging.getLogger(__name__)


@DatasetReader.register("coref")
class ConllCorefReader(DatasetReader):
    """
    Reads a single CoNLL-formatted file. This is the same file format as used in the
    :class:`~allennlp.data.dataset_readers.semantic_role_labelling.SrlReader`, but is preprocessed
    to dump all documents into a single file per train, dev and test split. See
    scripts/compile_coref_data.sh for more details of how to pre-process the Ontonotes 5.0 data
    into the correct format.

    Returns a `Dataset` where the `Instances` have four fields : `text`, a `TextField`
    containing the full document text, `spans`, a `ListField[SpanField]` of inclusive start and
    end indices for span candidates, and `metadata`, a `MetadataField` that stores the instance's
    original text. For data with gold cluster labels, we also include the original `clusters`
    (a list of list of index pairs) and a `SequenceLabelField` of cluster ids for every span
    candidate.

    # Parameters

    max_span_width : `int`, required.
        The maximum width of candidate spans to consider.
    token_indexers : `Dict[str, TokenIndexer]`, optional
        This is used to index the words in the document.  See :class:`TokenIndexer`.
        Default is `{"tokens": SingleIdTokenIndexer()}`.
    wordpiece_modeling_tokenizer: `PretrainedTransformerTokenizer`, optional (default = `None`)
        If not None, this dataset reader does subword tokenization using the supplied tokenizer
        and distribute the labels to the resulting wordpieces. All the modeling will be based on
        wordpieces. If this is set to `False` (default), the user is expected to use
        `PretrainedTransformerMismatchedIndexer` and `PretrainedTransformerMismatchedEmbedder`,
        and the modeling will be on the word-level.
    max_sentences: `int`, optional (default = `None`)
        The maximum number of sentences in each document to keep. By default keeps all sentences.
    remove_singleton_clusters : `bool`, optional (default = `False`)
        Some datasets contain clusters that are singletons (i.e. no coreferents). This option allows
        the removal of them. Ontonotes shouldn't have these, and this option should be used for
        testing only.
    """

    def __init__(
        self,
        max_span_width: int,
        token_indexers: Dict[str, TokenIndexer] = None,
        wordpiece_modeling_tokenizer: Optional[PretrainedTransformerTokenizer] = None,
        max_sentences: int = None,
        remove_singleton_clusters: bool = False,
        test_run: bool = False,
        pickle_path: str = None,
        srl: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._max_span_width = max_span_width
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._wordpiece_modeling_tokenizer = wordpiece_modeling_tokenizer
        self._max_sentences = max_sentences
        self._remove_singleton_clusters = remove_singleton_clusters
        self._test_run = test_run
        self._pickle_path = pickle_path
        self._srl = srl

    @overrides
    def _read(self, file_path: str):
        read_from_pickle = False
        if self._pickle_path is not None:
            if os.path.exists(self._pickle_path):
                read_from_pickle = True
                f = open(self._pickle_path, 'rb')
                instances = pickle.load(f)
                f.close()
                for instance in instances:
                    yield instance
        if not read_from_pickle:
            # if `file_path` is a URL, redirect to the cache
            file_path = cached_path(file_path)

            ontonotes_reader = Ontonotes()
            instances = []
            for sentences in ontonotes_reader.dataset_document_iterator(file_path):
                clusters: DefaultDict[int, List[Tuple[int, int]]] = collections.defaultdict(list)
                srl_frames = []

                total_tokens = 0
                for sentence in sentences:
                    for typed_span in sentence.coref_spans:
                        # Coref annotations are on a _per sentence_
                        # basis, so we need to adjust them to be relative
                        # to the length of the document.
                        span_id, (start, end) = typed_span
                        clusters[span_id].append((start + total_tokens, end + total_tokens))
                    for _, srl_tags in sentence.srl_frames:
                        predicate_index = None
                        for i in range(len(srl_tags)):
                            if srl_tags[i][-2:] == "-V":
                                predicate_index = i
                                break
                        if predicate_index is None:
                            print(srl_tags)
                            continue
                        # assert predicate_index is not None
                        predicate_index += total_tokens
                        current = None
                        current_type = None
                        arguments = []
                        argument_types = []
                        for i in range(len(srl_tags)):
                            if srl_tags[i] == "O" and current is not None:
                                arguments.append((current, i-1))
                                argument_types.append(current_type)
                                current = None
                                current_type = None
                            elif srl_tags[i][:2] == "B-":
                                if current is not None:
                                    arguments.append((current, i-1))
                                    argument_types.append(current_type)
                                current = i
                                current_type = srl_tags[i][2:]
                        if current is not None:
                            arguments.append((current, len(srl_tags)-1))
                            argument_types.append(current_type)
                        arguments = [(start+total_tokens, end+total_tokens, arg_type) for (start, end), arg_type in zip(arguments, argument_types)]
                        srl_frames.append((predicate_index, arguments))
                    total_tokens += len(sentence.words)

                instance = self.text_to_instance(sentences=[s.words for s in sentences], gold_clusters=list(clusters.values()), srl_frames=srl_frames)
                instances.append(instance)
                yield instance
                if self._test_run:
                    break
            if not self._test_run and self._pickle_path is not None:
                f = open(self._pickle_path, 'wb')
                pickle.dump(instances, f)
                f.close()

    @overrides
    def text_to_instance(
        self,  # type: ignore
        sentences: List[List[str]],
        words: Optional[List[str]] = None,
        gold_clusters: Optional[List[List[Tuple[int, int]]]] = None,
        srl_frames: Optional[List[Tuple[int, List[Tuple[int, int, str]]]]] = None,
    ) -> Instance:
        return make_coref_instance(
            sentences,
            self._token_indexers,
            self._max_span_width,
            words=words,
            gold_clusters=gold_clusters,
            srl_frames=srl_frames,
            include_srl=self._srl,
            wordpiece_modeling_tokenizer=self._wordpiece_modeling_tokenizer,
            max_sentences=self._max_sentences,
            remove_singleton_clusters=self._remove_singleton_clusters,
        )
