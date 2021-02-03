import logging
import math
from typing import Any, Dict, List, Tuple, cast
import os
import random
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import torch
import torch.nn.functional as F
from overrides import overrides

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.modules import FeedForward, GatedSum, ConditionalRandomField
from allennlp.modules.matrix_attention import BilinearMatrixAttention
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor, EndpointSpanExtractor
from allennlp.nn import util, InitializerApplicator
from allennlp.training.metrics import SpanBasedF1Measure, FBetaMeasure

from allennlp_models.coref.metrics.conll_coref_scores import ConllCorefScores
from allennlp_models.coref.metrics.mention_recall import MentionRecall
from allennlp_models.structured_prediction.metrics.e2e_srl_metric import E2eSrlMetric
from allennlp_models.structured_prediction.metrics.srl_eval_scorer import SrlEvalScorer
from allennlp_models.common.feedforward_layernorm import FeedForwardWithLayerNorm

from allennlp_models.structured_prediction.modules.batch_lpsmap import BatchLpsmap
from allennlp_models.structured_prediction.modules.lpsmap_loss import LpsmapLoss

from lpsmap import TorchFactorGraph, Xor, Budget, AtMostOne

import torch_struct

logger = logging.getLogger(__name__)


@Model.register("coref_srl")
class CoreferenceResolver(Model):
    """
    This `Model` implements the coreference resolution model described in
    [Higher-order Coreference Resolution with Coarse-to-fine Inference](https://arxiv.org/pdf/1804.05392.pdf)
    by Lee et al., 2018.
    The basic outline of this model is to get an embedded representation of each span in the
    document. These span representations are scored and used to prune away spans that are unlikely
    to occur in a coreference cluster. For the remaining spans, the model decides which antecedent
    span (if any) they are coreferent with. The resulting coreference links, after applying
    transitivity, imply a clustering of the spans in the document.

    # Parameters

    vocab : `Vocabulary`
    text_field_embedder : `TextFieldEmbedder`
        Used to embed the `text` `TextField` we get as input to the model.
    context_layer : `Seq2SeqEncoder`
        This layer incorporates contextual information for each word in the document.
    mention_feedforward : `FeedForward`
        This feedforward network is applied to the span representations which is then scored
        by a linear layer.
    antecedent_feedforward : `FeedForward`
        This feedforward network is applied to pairs of span representation, along with any
        pairwise features, which is then scored by a linear layer.
    feature_size : `int`
        The embedding size for all the embedded features, such as distances or span widths.
    max_span_width : `int`
        The maximum width of candidate spans.
    spans_per_word: `float`, required.
        A multiplier between zero and one which controls what percentage of candidate mention
        spans we retain with respect to the number of words in the document.
    max_antecedents: `int`, required.
        For each mention which survives the pruning stage, we consider this many antecedents.
    coarse_to_fine: `bool`, optional (default = `False`)
        Whether or not to apply the coarse-to-fine filtering.
    inference_order: `int`, optional (default = `1`)
        The number of inference orders. When greater than 1, the span representations are
        updated and coreference scores re-computed.
    lexical_dropout : `int`
        The probability of dropping out dimensions of the embedded text.
    initializer : `InitializerApplicator`, optional (default=`InitializerApplicator()`)
        Used to initialize the model parameters.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        context_layer: Seq2SeqEncoder,
        mention_feedforward: FeedForward,
        antecedent_feedforward: FeedForward,
        feature_size: int,
        max_span_width: int,
        spans_per_word: float,
        max_antecedents: int,
        coarse_to_fine: bool = False,
        inference_order: int = 1,
        lexical_dropout: float = 0.2,
        predict_srl: bool = False,
        predict_coref: bool = True,
        srl_predicate_candidates_per_word: float = 0.4,
        srl_predicate_feedforward1: FeedForward = None,
        srl_predicate_feedforward2: FeedForward = None,
        srl_argument_feedforward: FeedForward = None,
        srl_scorer: FeedForward = None,
        srl_predicate_scorer: FeedForward = None,
        srl_dropout: float = 0.1,
        srl_e2e: bool = True,
        srl_lpsmap: bool = False,
        srl_lpsmap_max_iter: int = 10,
        srl_lpsmap_overlapping_spans: bool = False,
        max_sentence_length: int = 210, # longest sentence in English Ontonotes training data
        predict_ner: bool = False,
        ner_sequence: bool = True,
        ner_tag_embedding_dim: int = 0,
        ner_train_with_constraints: bool = False,
        ner_sequence_sparsemax: bool = False,
        ner_feedforward: Seq2SeqEncoder = None,
        ner_scorer: FeedForward = None,
        language_masking_map: Dict[str, bool] = None,
        only_language_masking_map: Dict[str, bool] = None,
        consistency_map: Dict[str, bool] = None,
        consistency_with_supervised_loss: bool = True,
        consistency_loss_coefficient: float = 1.0,
        mention_score_loss: bool = False,
        load_weights_path: str = None,
        load_all_weights_path: str = None,
        non_span_rep_model: bool = False,
        start_feedforward1: FeedForwardWithLayerNorm = None,
        end_feedforward1: FeedForwardWithLayerNorm = None,
        start_feedforward2: FeedForwardWithLayerNorm = None,
        end_feedforward2: FeedForwardWithLayerNorm = None,
        consistency_epoch_threshold: int = 0,
        bug_fix: bool = False,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs
    ) -> None:
        super().__init__(vocab, **kwargs)

        self._text_field_embedder = text_field_embedder
        self._context_layer = context_layer
        self._non_span_rep_model = non_span_rep_model
        if non_span_rep_model:
            self._start_feedforward1 = start_feedforward1
            self._end_feedforward1 = end_feedforward1
            self._start_feedforward2 = start_feedforward2
            self._end_feedforward2 = end_feedforward2
            assert end_feedforward1.get_output_dim() == start_feedforward1.get_output_dim() == start_feedforward2.get_output_dim() == end_feedforward2.get_output_dim()
            hidden_size = start_feedforward1.get_output_dim()
            self._mention_scorer_start = torch.nn.Linear(hidden_size, 1)
            self._mention_scorer_end = torch.nn.Linear(hidden_size, 1)
            self._mention_scorer_bilinear = torch.nn.Linear(hidden_size, hidden_size) # BilinearMatrixAttention(hidden_size, hidden_size)
            self._antecedent_scorer_start_start = torch.nn.Linear(hidden_size, hidden_size) # BilinearMatrixAttention(hidden_size, hidden_size)
            self._antecedent_scorer_start_end = torch.nn.Linear(hidden_size, hidden_size) # BilinearMatrixAttention(hidden_size, hidden_size)
            self._antecedent_scorer_end_start = torch.nn.Linear(hidden_size, hidden_size) # BilinearMatrixAttention(hidden_size, hidden_size)
            self._antecedent_scorer_end_end = torch.nn.Linear(hidden_size, hidden_size) # BilinearMatrixAttention(hidden_size, hidden_size)
        else:
            self._mention_feedforward = TimeDistributed(mention_feedforward)
            self._mention_scorer = TimeDistributed(
                torch.nn.Linear(mention_feedforward.get_output_dim(), 1)
            )
            self._antecedent_feedforward = TimeDistributed(antecedent_feedforward)
            self._antecedent_scorer = TimeDistributed(
                torch.nn.Linear(antecedent_feedforward.get_output_dim(), 1)
            )

            self._endpoint_span_extractor = EndpointSpanExtractor(
                context_layer.get_output_dim(),
                combination="x,y",
                num_width_embeddings=max_span_width,
                span_width_embedding_dim=feature_size,
                bucket_widths=False,
            )
            self._attentive_span_extractor = SelfAttentiveSpanExtractor(
                input_dim=text_field_embedder.get_output_dim()+(ner_tag_embedding_dim if predict_ner and ner_sequence else 0)
            )
        self._bug_fix = bug_fix
        self._language_masking_map = language_masking_map
        self._only_language_masking_map = only_language_masking_map
        self._consistency_map = consistency_map
        self._mention_score_loss = mention_score_loss
        self._consistency_epoch_threshold = consistency_epoch_threshold
        self._consistency_with_supervised_loss = consistency_with_supervised_loss
        self._consistency_loss_coefficient = consistency_loss_coefficient

        self._predict_srl = predict_srl
        self._predict_coref = predict_coref
        self._predict_ner = predict_ner
        if predict_srl:
            if srl_predicate_feedforward1 is not None:
                self._srl_predicate_feedforward1 = TimeDistributed(srl_predicate_feedforward1)
            else:
                self._srl_predicate_feedforward1 = lambda x: x
            if srl_predicate_feedforward2 is not None:
                self._srl_predicate_feedforward2 = TimeDistributed(srl_predicate_feedforward2)
                predicate_embedding_size = srl_predicate_feedforward2.get_output_dim()
            else:
                self._srl_predicate_feedforward2 = lambda x: x
                predicate_embedding_size = context_layer.get_output_dim()
            if srl_argument_feedforward is not None:
                self._srl_argument_feedforward = TimeDistributed(srl_argument_feedforward)
            else:
                self._srl_argument_feedforward = lambda x: x
            if srl_e2e:
                label_namespace = "srl_labels"
                if non_span_rep_model:
                    self._argument_scorer_start = BilinearMatrixAttention(hidden_size, hidden_size, label_dim=self.vocab.get_vocab_size(label_namespace))
                    self._argument_scorer_end = BilinearMatrixAttention(hidden_size, hidden_size, label_dim=self.vocab.get_vocab_size(label_namespace))
            else:
                label_namespace = "srl_seq_labels"
            if srl_scorer is None:
                if srl_argument_feedforward is not None:
                    argument_embedding_size = srl_argument_feedforward.get_output_dim()
                else:
                    argument_embedding_size = mention_feedforward.get_input_dim()
                self._srl_scorer = TimeDistributed(
                    torch.nn.Linear(predicate_embedding_size+srl_argument_feedforward, self.vocab.get_vocab_size(namespace=label_namespace))
                )
            else:
                srl_scorer = torch.nn.Sequential(
                    srl_scorer,
                    torch.nn.Linear(srl_scorer.get_output_dim(), self.vocab.get_vocab_size(namespace=label_namespace)),
                )
                self._srl_scorer = TimeDistributed(srl_scorer)
            if srl_predicate_scorer is None:
                self._srl_predicate_scorer = TimeDistributed(
                    torch.nn.Linear(predicate_embedding_size, 1)
                )
            else:
                self._srl_predicate_scorer = TimeDistributed(srl_predicate_scorer)
            self._srl_dropout = torch.nn.Dropout(p=srl_dropout)
            self._predicate_candidates_per_word = srl_predicate_candidates_per_word
            self._srl_e2e = srl_e2e
            if srl_e2e:
                self._srl_metric = E2eSrlMetric()
            else:
                self._srl_metric = SrlEvalScorer(ignore_classes=["V"])
                label_encoding = "BIO"
                labels = self.vocab.get_index_to_token_vocabulary(label_namespace)
                constraints = allowed_transitions(label_encoding, labels)
                self.srl_crf = ConditionalRandomField(
                    self.vocab.get_vocab_size(label_namespace), constraints, include_start_end_transitions=True
                )
            self._lpsmap = srl_lpsmap
            self._lpsmap_max_iter = srl_lpsmap_max_iter
            self._lpsmap_overlapping_spans_srl = srl_lpsmap_overlapping_spans
            self._max_num_spans = int(spans_per_word*max_sentence_length)
            self._max_num_predicates = int(srl_predicate_candidates_per_word*max_sentence_length)
            if srl_lpsmap:
                self.constraint_types = []
                self.constraint_sizes = []
                self.budgets = []
                self.coefficients = []
                self.negated = []
                self.constraint_sets = []
                self.num_variables = self._max_num_spans*(1+self.vocab.get_vocab_size(namespace=label_namespace))
                # Exactly one label (possibly null) per predicate
                self.constraint_types.append("xor")
                self.constraint_sizes.append(1+self.vocab.get_vocab_size(namespace=label_namespace))
                self.budgets.append(1)
                self.negated.append([False for _ in range(self.constraint_sizes[-1])])
                self.coefficients.append([1 for _ in self.negated[-1]])
                self.constraint_sets.append([i for span in range(self._max_num_spans) for i in range(span*self.constraint_sizes[-1], (span+1)*self.constraint_sizes[-1])])
                # At most one argument for each core role
                self.constraint_types.append("budget")
                self.constraint_sizes.append(self._max_num_spans)
                self.budgets.append(1)
                self.negated.append([False for _ in range(self.constraint_sizes[-1])])
                self.coefficients.append([1 for _ in self.negated[-1]])
                label_vocab = self.vocab.get_token_to_index_vocabulary(label_namespace)
                self.core_role_labels = [self.vocab.get_token_index(role, namespace=label_namespace) for role in ["ARG0", "ARG1", "ARG2", "ARG3", "ARG4", "ARG5"] if role in label_vocab]
                self.constraint_sets.append([span*self.constraint_sizes[0]+label for label in self.core_role_labels for span in range(self._max_num_spans)])
                self._batch_lpsmap = None
        self._ner_sequence = ner_sequence
        if predict_ner:
            if ner_sequence:
                label_encoding = "BIO"
                label_namespace = "ner_seq_labels"
                labels = self.vocab.get_index_to_token_vocabulary(label_namespace)
                self._ner_tag_embedding_dim = ner_tag_embedding_dim
                self._ner_train_with_constraints = ner_train_with_constraints
                self._ner_metric = SpanBasedF1Measure(vocab, tag_namespace=label_namespace, label_encoding=label_encoding)
                constraints = allowed_transitions(label_encoding, labels)
                self.ner_crf = ConditionalRandomField(
                    self.vocab.get_vocab_size(label_namespace), constraints, include_start_end_transitions=True
                )
                self._ner_sequence_sparsemax = ner_sequence_sparsemax
                if ner_tag_embedding_dim > 0:
                    self._ner_tag_embeddings = torch.nn.Parameter(torch.zeros((len(labels), ner_tag_embedding_dim)))
                    self.ner_crf.transitions = torch.nn.Parameter(torch.Tensor(len(labels)+2, len(labels)+2))
                    self.ner_crf.reset_parameters()
                self._ner_feedforward = ner_feedforward
            else:
                label_namespace = "ner_span_labels"
                self._ner_metric = FBetaMeasure(average="macro")
            self._ner_scorer = torch.nn.Sequential(
                ner_scorer,
                torch.nn.Linear(ner_scorer.get_output_dim(), self.vocab.get_vocab_size(namespace=label_namespace)),
            )
            self._ner_scorer = TimeDistributed(self._ner_scorer)

        # 10 possible distance buckets.
        self._num_distance_buckets = 10
        self._distance_embedding = Embedding(
            embedding_dim=feature_size, num_embeddings=self._num_distance_buckets
        )

        self._max_span_width = max_span_width
        self._spans_per_word = spans_per_word
        self._max_antecedents = max_antecedents

        self._coarse_to_fine = coarse_to_fine
        if self._coarse_to_fine:
            self._coarse2fine_scorer = torch.nn.Linear(
                mention_feedforward.get_input_dim(), mention_feedforward.get_input_dim()
            )
        self._inference_order = inference_order
        if self._inference_order > 1:
            self._span_updating_gated_sum = GatedSum(mention_feedforward.get_input_dim())

        self._mention_recall = MentionRecall()
        self._conll_coref_scores = ConllCorefScores()
        if lexical_dropout > 0:
            self._lexical_dropout = torch.nn.Dropout(p=lexical_dropout)
        else:
            self._lexical_dropout = lambda x: x
        initializer(self)
        if load_weights_path is not None and len(load_weights_path) > 0:
            state = torch.load(load_weights_path)
            filtered_weights = {k.split("transformer_model.")[1]: state[k] for k in state if "transformer_model." in k}
            logger.info("Loading "+str(len(filtered_weights))+" transformer weights")
            embedder = list(self._text_field_embedder._token_embedders.values())[0]._matched_embedder
            embedder.transformer_model.load_state_dict(filtered_weights)
        if load_all_weights_path is not None and len(load_all_weights_path) > 0:
            state = torch.load(load_all_weights_path)
            self.load_state_dict(state)
        self.epoch = -1

    @overrides
    def forward(
        self,  # type: ignore
        text: TextFieldTensors,
        spans: torch.IntTensor,
        modified_text: TextFieldTensors = None,
        modified_spans: torch.IntTensor = None,
        span_labels: torch.IntTensor = None,
        span_loss_mask: torch.IntTensor = None,
        srl_labels: torch.LongTensor = None,
        srl_seq_labels: torch.LongTensor = None,
        srl_seq_indices: torch.LongTensor = None,
        srl_seq_predicates: torch.LongTensor = None,
        ner_seq_labels: torch.LongTensor = None,
        ner_span_labels: torch.LongTensor = None,
        word_span_mask: torch.IntTensor = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:

        """
        # Parameters

        text : `TextFieldTensors`, required.
            The output of a `TextField` representing the text of
            the document.
        spans : `torch.IntTensor`, required.
            A tensor of shape (batch_size, num_spans, 2), representing the inclusive start and end
            indices of candidate spans for mentions. Comes from a `ListField[SpanField]` of
            indices into the text of the document.
        span_labels : `torch.IntTensor`, optional (default = `None`).
            A tensor of shape (batch_size, num_spans), representing the cluster ids
            of each span, or -1 for those which do not appear in any clusters.
        metadata : `List[Dict[str, Any]]`, optional (default = `None`).
            A metadata dictionary for each instance in the batch. We use the "original_text" and "clusters" keys
            from this dictionary, which respectively have the original text and the annotated gold coreference
            clusters for that instance.

        # Returns

        An output dictionary consisting of:

        top_spans : `torch.IntTensor`
            A tensor of shape `(batch_size, num_spans_to_keep, 2)` representing
            the start and end word indices of the top spans that survived the pruning stage.
        antecedent_indices : `torch.IntTensor`
            A tensor of shape `(num_spans_to_keep, max_antecedents)` representing for each top span
            the index (with respect to top_spans) of the possible antecedents the model considered.
        predicted_antecedents : `torch.IntTensor`
            A tensor of shape `(batch_size, num_spans_to_keep)` representing, for each top span, the
            index (with respect to antecedent_indices) of the most likely antecedent. -1 means there
            was no predicted link.
        loss : `torch.FloatTensor`, optional
            A scalar loss to be optimised.
        """
        loss = torch.tensor(0., requires_grad=True).to(spans.device)
        # if metadata is not None and modified_text is None:
        #     logger.info(metadata[0]["language"])
        if metadata is not None and "language" in metadata[0] and self._language_masking_map is not None and self._language_masking_map[metadata[0]["language"]]:
            key = self._text_field_embedder._ordered_embedder_keys[0]
            input_tensors = text[key]
            embedder = getattr(self._text_field_embedder, "token_embedder_"+key)
            _, masked_lm_loss = embedder(**input_tensors, masked_lm=[self._language_masking_map[metadata[i]["language"]] for i in range(len(metadata))])
            num_masking_languages = sum([1 if self._language_masking_map[lang] else 0 for lang in self._language_masking_map])
            loss += masked_lm_loss/float(num_masking_languages)
            if self._only_language_masking_map is not None and self._only_language_masking_map[metadata[0]["language"]]:
                outputs = {"loss": loss}
                return outputs
            # elif metadata is not None and "language" in metadata[0] and self._consistency_map is not None and self._consistency_map[metadata[0]["language"]] and modified_text is not None:
        elif modified_text is not None and (self._consistency_epoch_threshold is None or self.epoch >= self._consistency_epoch_threshold):
            assert modified_spans is not None
            assert self._predict_coref
            assert self._non_span_rep_model
            training_mode = self.training
            if self.training:
                full_text_output_dict = self.make_output_human_readable(self.forward(text=text, spans=spans.clone(), span_labels=span_labels, metadata=metadata))
            else:
                full_text_output_dict = self.make_output_human_readable(self.forward(text=text, spans=spans.clone()))

            if (self._consistency_with_supervised_loss or span_labels is None): # and self.training:
                # logger.info('here')
                if "removed_text_start" in metadata[0] and "removed_text_end" in metadata[0]:
                    removed_text_start = metadata[0]["removed_text_start"]
                    removed_text_end = metadata[0]["removed_text_end"]
                else:
                    removed_text_start = -1
                    removed_text_end = -1
                removed_text_length = removed_text_end-removed_text_start
                original_span_index_map = metadata[0]["span_index_map"]
                original_clusters = []
                exclude_mentions = set()
                original_span_labels = -1*torch.ones_like(spans[:,:,0])
                original_span_mask = torch.ones_like(spans[:,:,0]).bool()
                for cluster in full_text_output_dict["clusters"][0]:
                    include = True
                    for mention in cluster:
                        if mention[0] >= removed_text_start and mention[1] < removed_text_end:
                            include = False
                            break
                    if not include:
                        for mention in cluster:
                            exclude_mentions.add(tuple(mention))
                            original_span_mask[0,original_span_index_map[tuple(mention)]] = False
                    else:
                        original_clusters.append(tuple((m[0]-removed_text_length, m[1]-removed_text_length) if m[0] >= removed_text_start else tuple(m) for m in cluster))
                        for m in cluster:
                            original_span_labels[0,original_span_index_map[tuple(m)]] = len(original_clusters)
                for i in range(spans.shape[1]):
                    if spans[0,i,0].item() >= removed_text_start and spans[0,i,1].item() < removed_text_end:
                        original_span_mask[0,i] = False
                assert len(metadata[0]["modified_span_indices"]) == modified_spans.shape[1]
                assert (modified_spans == torch.where(spans[:,metadata[0]["modified_span_indices"],:] >= removed_text_start, spans[:,metadata[0]["modified_span_indices"],:]-removed_text_length, spans[:,metadata[0]["modified_span_indices"],:])).all().item()
                original_span_labels_mapped = original_span_labels[:,metadata[0]["modified_span_indices"]]
                if self.training:
                    modified_text_outputs = self.forward(text=modified_text, spans=modified_spans)
                else:
                    modified_metadata = [{key: m[key] for key in m} for m in metadata]
                    modified_metadata[0]["clusters"] = original_clusters
                    modified_text_outputs = self.forward(text=modified_text, spans=modified_spans, span_labels=-1*torch.ones_like(modified_spans[:,:,0]), metadata=modified_metadata)
                modified_spans_masked = torch.where((original_span_mask[:,metadata[0]["modified_span_indices"]] > 0).unsqueeze(-1).repeat(1, 1, 2), modified_spans, -1*torch.ones_like(modified_spans))
                mention_mask = modified_spans_masked[0,:,0] > -1
                full_text_span_mention_distributions = torch.cat((torch.zeros_like(full_text_output_dict["span_mention_scores"][0,metadata[0]["modified_span_indices"]]).unsqueeze(-1),
                    full_text_output_dict["span_mention_scores"][0,metadata[0]["modified_span_indices"]].unsqueeze(-1)), dim=1).log_softmax(dim=1)[mention_mask,:]
                modified_text_span_mention_distributions = torch.cat((torch.zeros_like(modified_text_outputs["span_mention_scores"][0,:].unsqueeze(-1)),
                    modified_text_outputs["span_mention_scores"][0,:].unsqueeze(-1)), dim=1).log_softmax(dim=1)[mention_mask,:]
                if self._mention_score_loss:
                    mention_scores_loss = (torch.nn.BCEWithLogitsLoss()(modified_text_outputs["span_mention_scores"][0,mention_mask], full_text_output_dict["span_mention_scores"][0,metadata[0]["modified_span_indices"]][mention_mask].sigmoid())+torch.nn.BCEWithLogitsLoss()(full_text_output_dict["span_mention_scores"][0,metadata[0]["modified_span_indices"]][mention_mask], modified_text_outputs["span_mention_scores"][0,mention_mask].sigmoid()))/2
                    # logger.info(mention_scores_loss)
                    # mention_scores_loss = (torch.nn.KLDivLoss(reduction="batchmean")(modified_text_span_mention_distributions, full_text_span_mention_distributions)+torch.nn.KLDivLoss(reduction="batchmean")(full_text_span_mention_distributions, modified_text_span_mention_distributions))/2
                    # if mention_scores_loss.item() <= full_text_output_dict["loss"].item():
                    full_text_output_dict["loss"] += mention_scores_loss
                else:
                    # Shape: (1, original_num_spans)
                    original_modified_span_indices_bool_mask = torch.zeros_like(original_span_mask)
                    original_modified_span_indices_bool_mask[:,metadata[0]["modified_span_indices"]] = True
                    # Shape: (1, original_num_spans)
                    original_top_span_indices_bool_mask = torch.zeros_like(original_span_mask)
                    # Shape: (1, original_num_top_spans)
                    original_top_span_indices_bool_mask[:,full_text_output_dict["top_span_indices"][0,:]] = True
                    # Shape: (1, original_num_spans)
                    original_top_span_indices_bool_mask &= (original_modified_span_indices_bool_mask & original_span_mask)
                    # Shape: (1, modified_num_spans)
                    modified_top_span_indices_bool_mask = torch.zeros_like(original_span_mask[:,metadata[0]["modified_span_indices"]])
                    # Shape: (1, modified_num_top_spans)
                    modified_top_span_indices_bool_mask[:,modified_text_outputs["top_span_indices"][0,:]] = True
                    modified_top_span_indices_bool_mask &= mention_mask.unsqueeze(0)
                    modified_top_span_indices_bool_mask &= original_top_span_indices_bool_mask[:,metadata[0]["modified_span_indices"]]
                    original_top_span_indices_bool_mask[:,metadata[0]["modified_span_indices"]] &= modified_top_span_indices_bool_mask
                    if modified_top_span_indices_bool_mask.any().item():
                        original_row_indices = original_top_span_indices_bool_mask[0,full_text_output_dict["top_span_indices"][0,:]]
                        original_column_indices = torch.arange(1+full_text_output_dict["top_span_indices"].numel()).to(spans.device)[[True]+original_top_span_indices_bool_mask[0,full_text_output_dict["top_span_indices"][0,:]].tolist()]
                        # Shape: (num_pruned_spans, num_pruned_spans+1)
                        original_pruned_coreference_scores = torch.gather(full_text_output_dict["coreference_scores"][0][original_row_indices], 1, original_column_indices.unsqueeze(0).repeat(original_row_indices.long().sum().item(), 1))
                        original_pruned_antecedent_mask = torch.gather(full_text_output_dict["antecedent_mask"][0][original_row_indices], dim=1, index=original_column_indices[1:].unsqueeze(0).repeat(original_row_indices.long().sum().item(), 1)-1)
                        original_pruned_antecedent_mask = torch.cat((torch.ones_like(original_pruned_antecedent_mask[:,:1]), original_pruned_antecedent_mask), dim=1)
                        # original_pruned_coreference_log_probs = torch.where(original_pruned_antecedent_mask, util.masked_log_softmax(original_pruned_coreference_scores, original_pruned_antecedent_mask), torch.zeros_like(original_pruned_coreference_scores))
                        original_pruned_coreference_log_probs = util.replace_masked_values(
                            util.masked_log_softmax(original_pruned_coreference_scores, original_pruned_antecedent_mask),
                            original_pruned_antecedent_mask,
                            util.min_value_of_dtype(original_pruned_coreference_scores.dtype)
                        )
                        original_pruned_coreference_probs = util.masked_softmax(original_pruned_coreference_scores, original_pruned_antecedent_mask)
                        modified_row_indices = modified_top_span_indices_bool_mask[0,modified_text_outputs["top_span_indices"][0,:]]
                        modified_column_indices = torch.arange(1+modified_text_outputs["top_span_indices"].numel()).to(spans.device)[[True]+modified_top_span_indices_bool_mask[0,modified_text_outputs["top_span_indices"][0,:]].tolist()]
                        # Shape: (num_pruned_spans, num_pruned_spans+1)
                        modified_pruned_coreference_scores = torch.gather(modified_text_outputs["coreference_scores"][0][modified_row_indices], 1, modified_column_indices.unsqueeze(0).repeat(modified_row_indices.long().sum().item(), 1))
                        modified_pruned_antecedent_mask = torch.gather(modified_text_outputs["antecedent_mask"][0][modified_row_indices], dim=1, index=modified_column_indices[1:].unsqueeze(0).repeat(modified_row_indices.long().sum().item(), 1)-1)
                        modified_pruned_antecedent_mask = torch.cat((torch.ones_like(modified_pruned_antecedent_mask[:,:1]), modified_pruned_antecedent_mask), dim=1)
                        # modified_pruned_coreference_log_probs = torch.where(modified_pruned_antecedent_mask, util.masked_log_softmax(modified_pruned_coreference_scores, modified_pruned_antecedent_mask), torch.zeros_like(modified_pruned_coreference_scores))
                        modified_pruned_coreference_log_probs = util.replace_masked_values(
                            util.masked_log_softmax(modified_pruned_coreference_scores, modified_pruned_antecedent_mask),
                            modified_pruned_antecedent_mask,
                            util.min_value_of_dtype(modified_pruned_coreference_scores.dtype)
                        )
                        modified_pruned_coreference_probs = util.masked_softmax(modified_pruned_coreference_scores, modified_pruned_antecedent_mask)
                        # antecedent_loss = 0.5*(-original_pruned_coreference_probs*modified_pruned_coreference_log_probs-modified_pruned_coreference_probs*original_pruned_coreference_log_probs).sum(1).mean()
                        antecedent_loss = 0.5*(torch.nn.KLDivLoss(reduction="mean")(original_pruned_coreference_log_probs, modified_pruned_coreference_probs)+torch.nn.KLDivLoss(reduction="mean")(modified_pruned_coreference_log_probs, original_pruned_coreference_probs))
                        # logger.info('A '+str(original_pruned_coreference_log_probs))
                        # logger.info('B '+str(modified_pruned_coreference_log_probs))
                        # logger.info('C '+str(antecedent_loss))
                        # antecedent_loss = antecedent_loss.sum()/original_pruned_coreference_log_probs.shape[0]
                        # logger.info(str(antecedent_loss))
                        # if antecedent_loss.item() <= full_text_output_dict["loss"].item():
                        full_text_output_dict["loss"] += antecedent_loss*self._consistency_loss_coefficient
                        assert (torch.where(full_text_output_dict["top_spans"][0,original_row_indices] >= removed_text_start, full_text_output_dict["top_spans"][0,original_row_indices]-removed_text_length, full_text_output_dict["top_spans"][0,original_row_indices]) == modified_text_outputs["top_spans"][0,modified_row_indices]).all().item()
                        assert (torch.where(full_text_output_dict["top_spans"][0,original_column_indices[1:]-1] >= removed_text_start, full_text_output_dict["top_spans"][0,original_column_indices[1:]-1]-removed_text_length, full_text_output_dict["top_spans"][0,original_column_indices[1:]-1]) == modified_text_outputs["top_spans"][0,modified_column_indices[1:]-1]).all().item()

                """modified_text_loss = modified_text_outputs_grad["loss"]
                loss = modified_text_loss"""

            return full_text_output_dict

        # Shape: (batch_size, document_length, embedding_size)
        text_embeddings = self._lexical_dropout(self._text_field_embedder(text))
        loss += 0.*text_embeddings.sum()

        batch_size = spans.size(0)
        document_length = text_embeddings.size(1)
        num_spans = spans.size(1)

        # Shape: (batch_size, document_length)
        text_mask = util.get_text_field_mask(text)

        # Shape: (batch_size, num_spans)
        span_mask = (spans[:, :, 0] >= 0).squeeze(-1)
        if span_loss_mask is None:
            span_loss_mask = torch.ones_like(span_mask)

        # SpanFields return -1 when they are used as padding. As we do
        # some comparisons based on span widths when we attend over the
        # span representations that we generate from these indices, we
        # need them to be <= 0. This is only relevant in edge cases where
        # the number of spans we consider after the pruning stage is >= the
        # total number of spans, because in this case, it is possible we might
        # consider a masked span.
        # Shape: (batch_size, num_spans, 2)
        spans = F.relu(spans.float()).long()

        # Shape: (batch_size, document_length, encoding_dim)
        contextualized_embeddings = self._context_layer(text_embeddings, text_mask)

        if self._predict_ner and self._ner_sequence and self._ner_tag_embedding_dim == 0:
            classifier_outputs = self._ner_scorer(contextualized_embeddings)
            if ner_seq_labels is not None:
                loss += -self.ner_crf(classifier_outputs, ner_seq_labels, text_mask)/text_mask.sum().float()
                if not self.training:
                    viterbi_output = self.ner_crf.viterbi_tags(
                        classifier_outputs,
                        text_mask,
                        top_k=1
                    )
                    predicted_tags = cast(List[List[int]], [x[0][0] for x in viterbi_output])
                    # Represent viterbi tags as "class probabilities" that we can
                    # feed into the metrics
                    class_probabilities = classifier_outputs * 0.0
                    for i, instance_tags in enumerate(predicted_tags):
                        for j, tag_id in enumerate(instance_tags):
                            class_probabilities[i, j, tag_id] = 1
                    self._ner_metric(class_probabilities, ner_seq_labels, text_mask)
                    predicted_string_tags = [[self.vocab.get_token_from_index(i, namespace="ner_seq_labels") for i in lst] for lst in predicted_tags]
                    output_dict["ner_predictions"] = predicted_string_tags
                    if metadata is not None:
                        output_dict["ner_seq_labels"] = [x["ner_seq_labels"] for x in metadata]
        elif self._predict_ner and self._ner_sequence and self._ner_tag_embedding_dim > 0:
            # Shape: (batch_size, document_length, num_tags)
            scores = self._ner_scorer(contextualized_embeddings)
            # Shape: (batch_size, document_length+1, num_tags)
            crf_input_scores = torch.cat((scores, torch.zeros_like(scores[:,:1,:])), dim=1)
            # Shape: (batch_size, document_length+1, num_tags+2, 1)
            crf_input_scores = torch.cat((crf_input_scores, torch.zeros_like(crf_input_scores[:,:,:1]).repeat(1, 1, 2)), dim=2).unsqueeze(-1)
            # Shape: (num_tags+2, num_tags+2)
            # transition_scores = torch.zeros((scores.shape[-1]+2, scores.shape[-1]+2)).to(scores.device)
            transition_scores = self.get_ner_transition_scores(include_constraints=((not self.training) or self._ner_train_with_constraints))
            # Shape: (batch_size, document_length+1, num_tags+2, num_tags+2)
            crf_input_scores = crf_input_scores.repeat(1, 1, 1, transition_scores.shape[1])+transition_scores.view(1, 1, transition_scores.shape[0], transition_scores.shape[1])
            dist = torch_struct.LinearChainCRF(crf_input_scores, lengths=text_mask.sum(1)+2)
            if self._ner_sequence_sparsemax:
                with torch.enable_grad():
                    class_probabilities = dist._struct(torch_struct.SparseMaxSemiring).marginals(dist.log_potentials, dist.lengths).sum(3)[:,:-1,:-2]
            elif self.training:
                # Shape: (batch_size, document_length, num_tags)
                class_probabilities = dist.marginals.sum(3)[:,:-1,:-2]
            else:
                # Shape: (batch_size, document_length, num_tags)
                class_probabilities = dist.argmax.sum(3)[:,:-1,:-2]
            # Shape: (batch_size, document_length, num_tags, tag_embedding_size)
            tag_embeddings = class_probabilities.unsqueeze(-1)*self._ner_tag_embeddings.unsqueeze(0).unsqueeze(0)
            # Shape: (batch_size, document_length, tag_embedding_size)
            tag_embeddings = tag_embeddings.sum(2)
            # Shape: (batch_size, document_length, text_embedding_size+tag_embedding_size)
            text_embeddings = torch.cat((text_embeddings, tag_embeddings), dim=2)
            if self._ner_feedforward is not None:
                text_embeddings = self._ner_feedforward(text_embeddings)
            if ner_seq_labels is not None:
                num_tags = self.vocab.get_vocab_size("ner_seq_labels")
                start_tag = num_tags
                end_tag = num_tags + 1
                # Shape: (batch_size, document_length+2)
                crf_labels = torch.cat((torch.ones_like(ner_seq_labels[:,:1])*start_tag,
                                        ner_seq_labels,
                                        torch.ones_like(ner_seq_labels[:,:1])*end_tag), dim=1)
                # Shape: (batch_size, document_length+1, num_tags+2, num_tags+2)
                crf_labels = torch_struct.LinearChainCRF.struct.to_parts(crf_labels, num_tags+2, lengths=text_mask.sum(1)+2)
                loss += -dist.log_prob(crf_labels).sum()/(text_mask.sum()+2)
                # loss += -self.ner_crf(scores, ner_seq_labels, text_mask)/text_mask.sum().float()
                # if not self.training:
                self._ner_metric(class_probabilities, ner_seq_labels, text_mask)

        # Prune based on mention scores.
        num_spans_to_keep = int(math.floor(self._spans_per_word * document_length))
        num_spans_to_keep = max(min(num_spans_to_keep, num_spans), 1)

        if self._non_span_rep_model:
            # Shape: (batch_size, document_length, embedding_size)
            start_representations = self._start_feedforward1(contextualized_embeddings)
            end_representations = self._end_feedforward1(contextualized_embeddings)
            # Shape: (batch_size, document_length)
            span_start_scores = self._mention_scorer_start(start_representations).squeeze(-1)
            span_end_scores = self._mention_scorer_end(end_representations).squeeze(-1)
            # Shape: (batch_size, document_length, document_length)
            """span_bilinear_scores = self._mention_scorer_bilinear(
                    start_representations,
                    end_representations
            ).view(batch_size, document_length, document_length)"""
            span_bilinear_scores = torch.matmul(
                self._mention_scorer_bilinear(start_representations),
                end_representations.permute(0, 2, 1),
            ).view(batch_size, document_length, document_length)
            # Shape: (batch_size, num_spans)
            span_mention_start_scores = torch.gather(span_start_scores, 1, spans[:,:,0])
            span_mention_end_scores = torch.gather(span_end_scores, 1, spans[:,:,1])
            batch_range = torch.arange(batch_size).to(spans.device).unsqueeze(-1).repeat(1, spans.shape[1])
            span_mention_bilinear_scores = span_bilinear_scores[
                batch_range.view(-1),
                spans[:,:,0].contiguous().view(-1),
                spans[:,:,1].contiguous().view(-1)
            ].view(batch_size, num_spans)
            span_mention_scores = span_mention_start_scores+span_mention_end_scores+span_mention_bilinear_scores
        else:
            # Shape: (batch_size, num_spans, 2 * encoding_dim + feature_size)
            endpoint_span_embeddings = self._endpoint_span_extractor(contextualized_embeddings, spans)
            # Shape: (batch_size, num_spans, emebedding_size)
            attended_span_embeddings = self._attentive_span_extractor(text_embeddings, spans)

            # Shape: (batch_size, num_spans, emebedding_size + 2 * encoding_dim + feature_size)
            span_embeddings = torch.cat([endpoint_span_embeddings, attended_span_embeddings], -1)

            transformed_span_embeddings = self._mention_feedforward(span_embeddings)

            # Shape: (batch_size, num_spans)
            span_mention_scores = self._mention_scorer(
                transformed_span_embeddings
            ).squeeze(-1)
        # Shape: (batch_size, num_spans) for all 3 tensors
        top_span_mention_scores, top_span_mask, top_span_indices = util.masked_topk(
            span_mention_scores, span_mask, num_spans_to_keep
        )

        # Shape: (batch_size * num_spans_to_keep)
        # torch.index_select only accepts 1D indices, but here
        # we need to select spans for each element in the batch.
        # This reformats the indices to take into account their
        # index into the batch. We precompute this here to make
        # the multiple calls to util.batched_index_select below more efficient.
        flat_top_span_indices = util.flatten_and_batch_shift_indices(top_span_indices, num_spans)

        # Compute final predictions for which spans to consider as mentions.
        # Shape: (batch_size, num_spans_to_keep, 2)
        top_spans = util.batched_index_select(spans, top_span_indices, flat_top_span_indices)
        if not self._non_span_rep_model:
            # Shape: (batch_size, num_spans_to_keep, embedding_size)
            top_span_embeddings = util.batched_index_select(
                span_embeddings, top_span_indices, flat_top_span_indices
            )

        output_dict = {
            "top_spans": top_spans,
            "all_spans": spans,
            "top_span_indices": top_span_indices,
            "span_mention_scores": span_mention_scores
        }

        if self._predict_srl and self._srl_e2e:
            # word_span_mask = word_span_mask.to_dense()
            if srl_labels is not None:
                srl_labels = srl_labels.to_dense()
            num_labels = self.vocab.get_vocab_size("srl_labels")
            # Shape: (batch_size, document_length, embedding_size)
            predicate_embeddings = self._srl_predicate_feedforward1(contextualized_embeddings)
            # Shape: (batch_size, document_length)
            predicate_candidate_scores = self._srl_predicate_scorer(predicate_embeddings).view(batch_size, document_length)
            num_predicate_candidates_to_keep = int(math.floor(self._predicate_candidates_per_word * document_length))
            num_predicate_candidates_to_keep = max(min(num_predicate_candidates_to_keep, document_length), 1)
            # Shape: (batch_size, num_predicate_candidates_to_keep) for all 3 tensors
            top_predicate_scores, top_predicate_mask, top_predicate_indices = util.masked_topk(
                predicate_candidate_scores, text_mask, num_predicate_candidates_to_keep
            )

            # Shape: (batch_size*num_predicate_candidates_to_keep)
            flat_top_predicate_indices = util.flatten_and_batch_shift_indices(top_predicate_indices, document_length)
            predicate_embeddings = self._srl_predicate_feedforward2(contextualized_embeddings)
            # Shape: (batch_size, num_predicate_candidates_to_keep, embedding_size)
            top_predicate_embeddings = util.batched_index_select(
                predicate_embeddings, top_predicate_indices, flat_top_predicate_indices
            )

            if self._non_span_rep_model:
                start_representations = self._start_feedforward2(contextualized_embeddings)
                end_representations = self._end_feedforward2(contextualized_embeddings)
                # Shape: (batch_size, num_predicate_candidates_to_keep, document_length, num_argument_types)
                argument_scores_start = self._argument_scorer_start(top_predicate_embeddings, start_representations).permute(0, 2, 3, 1)
                argument_scores_end = self._argument_scorer_end(top_predicate_embeddings, end_representations).permute(0, 2, 3, 1)
                # Shape: (batch_size, num_predicate_candidates_to_keep, num_spans_to_keep)
                top_argument_indices = torch.arange(num_spans_to_keep).to(spans.device).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_predicate_candidates_to_keep, 1)
                top_argument_mask = top_span_mask.unsqueeze(1).repeat(1, num_predicate_candidates_to_keep, 1)
                # Shape: (batch_size, num_predicate_candidates_to_keep, num_spans_to_keep)
                argument_mention_scores = top_span_mention_scores.unsqueeze(1).repeat(1, num_predicate_candidates_to_keep, 1)
                argument_starts = top_spans[:,:,0].unsqueeze(1).unsqueeze(-1).repeat(1, num_predicate_candidates_to_keep, 1, num_labels)
                argument_ends = top_spans[:,:,1].unsqueeze(1).unsqueeze(-1).repeat(1, num_predicate_candidates_to_keep, 1, num_labels)
                batch_range = torch.arange(batch_size).unsqueeze(-1).repeat(1, num_predicate_candidates_to_keep*num_spans_to_keep*num_labels).view(-1)
                predicate_range = torch.arange(num_predicate_candidates_to_keep).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(batch_size, 1, num_spans_to_keep, num_labels).view(-1)
                label_range = torch.arange(num_labels).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_predicate_candidates_to_keep, num_spans_to_keep, 1).view(-1)
                # Shape: (batch_size, num_predicate_candidates_to_keep, num_spans_to_keep, num_argument_types)
                argument_scores_start = argument_scores_start[
                    batch_range,
                    predicate_range,
                    argument_starts.view(-1),
                    label_range
                ].view(batch_size, num_predicate_candidates_to_keep, num_spans_to_keep, num_labels)
                argument_scores_end = argument_scores_end[
                    batch_range,
                    predicate_range,
                    argument_ends.view(-1),
                    label_range
                ].view(batch_size, num_predicate_candidates_to_keep, num_spans_to_keep, num_labels)
                # Shape: (batch_size, num_predicate_candidates_to_keep, 1, 1)
                top_predicate_scores = util.replace_masked_values(
                    top_predicate_scores, top_predicate_mask, util.min_value_of_dtype(top_predicate_scores.dtype)
                ).unsqueeze(-1).unsqueeze(-1)
                # Shape: (batch_size, 1, num_spans_to_keep, 1)
                top_span_argument_scores = util.replace_masked_values(
                    top_span_mention_scores, top_span_mask, util.min_value_of_dtype(top_span_mention_scores.dtype)
                ).unsqueeze(1).unsqueeze(-1)

                # Shape: (batch_size, num_predicate_candidates_to_keep, num_spans_to_keep, num_labels)
                argument_scores = top_predicate_scores+top_span_argument_scores+argument_scores_start+argument_scores_end
                # Shape: (batch_size, num_predicate_candidates_to_keep, num_spans_to_keep, num_labels+1)
                argument_scores = torch.cat((torch.zeros_like(argument_scores[:,:,:,:1]), argument_scores), dim=-1)
            else:
                # Shape: (batch_size, num_spans_to_keep, embedding_size)
                argument_representations = self._srl_argument_feedforward(top_span_embeddings)
                # Shape: (batch_size, num_predicate_candidates_to_keep, num_spans_to_keep, embedding_size)
                argument_representations = argument_representations.unsqueeze(1).repeat(1, num_predicate_candidates_to_keep, 1, 1)
                # Shape: (batch_size, num_predicate_candidates_to_keep, num_spans_to_keep, embedding_size)
                top_predicate_embeddings = top_predicate_embeddings.unsqueeze(2).repeat(1, 1, num_spans_to_keep, 1)
                # Shape: (batch_size, num_predicate_candidates_to_keep, num_spans_to_keep, predicate_embedding_size+argument_embedding_size)
                predicate_argument_representations = torch.cat((top_predicate_embeddings, argument_representations), dim=-1)
                # Shape: (batch_size, 1, num_spans_to_keep, 1)
                top_span_argument_scores = util.replace_masked_values(
                    top_span_mention_scores, top_span_mask, util.min_value_of_dtype(top_span_mention_scores.dtype)
                ).unsqueeze(1).unsqueeze(-1)
                # Shape: (batch_size, num_predicate_candidates_to_keep, 1, 1)
                top_predicate_scores = util.replace_masked_values(
                    top_predicate_scores, top_predicate_mask, util.min_value_of_dtype(top_predicate_scores.dtype)
                ).unsqueeze(-1).unsqueeze(-1)
                # Shape: (batch_size, num_predicate_candidates_to_keep, num_spans_to_keep, num_labels)
                argument_scores = self._srl_scorer(predicate_argument_representations)+top_span_argument_scores+top_predicate_scores
                # Shape: (batch_size, num_predicate_candidates_to_keep, num_spans_to_keep, num_labels+1)
                argument_scores = torch.cat((torch.zeros_like(argument_scores[:,:,:,:1]), argument_scores), dim=-1)
            # Shape: (batch_size, num_predicate_candidates_to_keep, num_spans_to_keep)
            combined_mask = (top_predicate_mask.unsqueeze(-1).repeat(1, 1, num_spans_to_keep) & top_span_mask.unsqueeze(1).repeat(1, num_predicate_candidates_to_keep, 1))
            # Shape: (batch_size, num_predicate_candidates_to_keep, num_spans_to_keep, num_labels+1)
            argument_scores = util.replace_masked_values(
                argument_scores, combined_mask.unsqueeze(-1).repeat(1, 1, 1, num_labels+1), util.min_value_of_dtype(argument_scores.dtype)
            )
            if self._lpsmap:
                """if num_spans_to_keep < self._max_num_spans:
                    # Shape: (batch_size, num_predicate_candidates_to_keep, max_num_spans, num_labels+1)
                    argument_scores = torch.cat((argument_scores, torch.ones_like(argument_scores[:,:,:1,:]).repeat(1, 1, self._max_num_spans-num_spans_to_keep, 1)*util.min_value_of_dtype(argument_scores.dtype)), dim=2)
                if num_predicate_candidates_to_keep < self._max_num_predicates:
                    # Shape: (batch_size, max_num_predicates, max_num_spans, num_labels+1)
                    argument_scores = torch.cat((argument_scores, torch.ones_like(argument_scores[:,:1,:,:]).repeat(1, self._max_num_predicates-num_predicate_candidates_to_keep, 1, 1)*util.min_value_of_dtype(argument_scores.dtype)), dim=1)
                if self._batch_lpsmap is None:
                    self._batch_lpsmap = BatchLpsmap(self.num_variables, self.constraint_types, self.constraint_sizes, self.constraint_sets, self.budgets, self.negated, self.coefficients, argument_scores.device, batch_size*self._max_num_predicates, max_iter=self._lpsmap_max_iter)
                with torch.no_grad():
                    argument_probs = self._batch_lpsmap(argument_scores.view(batch_size*self._max_num_predicates, -1)).view(batch_size, self._max_num_predicates, self._max_num_spans, num_labels+1)
                argument_scores = argument_scores[:,:num_predicate_candidates_to_keep,:num_spans_to_keep,:].contiguous()
                argument_probs = argument_probs[:,:num_predicate_candidates_to_keep,:num_spans_to_keep,:].contiguous()"""
                word_span_mask = word_span_mask.to_dense()
                # Shape: (batch_size, document_length, num_spans_to_keep)
                word_span_mask = torch.gather(word_span_mask, 2, top_span_indices.unsqueeze(1).repeat(1, word_span_mask.shape[1], 1))
                with torch.no_grad():
                    argument_probs = []
                    for i in range(batch_size):
                        results = []
                        for j in range(num_predicate_candidates_to_keep):
                            fg = TorchFactorGraph()
                            x = fg.variable_from(argument_scores[i,j,:,:].cpu())
                            for s in range(num_spans_to_keep):
                                fg.add(Xor(x[s,:]))
                            label_vocab = self.vocab.get_token_to_index_vocabulary()
                            for l in self.core_role_labels:
                                fg.add(AtMostOne(x[:,l+1]))
                            if self._lpsmap_overlapping_spans_srl:
                                for word in range(document_length):
                                    spans_including_word = word_span_mask[i,word,:].nonzero().tolist()
                                    if len(spans_including_word) > 1:
                                        fg.add(AtMostOne(x[spans_including_word,1:]))
                            fg.solve(max_iter=self._lpsmap_max_iter)
                            results.append(x.value.to(argument_scores.device))
                        argument_probs.append(torch.stack(results))
                    argument_probs = torch.stack(argument_probs)
            else:
                argument_probs = argument_scores.softmax(dim=-1)
            predicted_tags = [[] for _ in range(batch_size)]
            predicted_indices = ((argument_probs.argmax(-1) > 0) & combined_mask).nonzero()
            for i in range(predicted_indices.shape[0]):
                batch_index, predicate_index, span_index = predicted_indices[i,:].tolist()
                predicted_tags[batch_index].append((top_predicate_indices[batch_index,predicate_index].item(), (top_spans[batch_index,span_index,0].item(), top_spans[batch_index,span_index,1].item(), self.vocab.get_token_from_index(argument_probs[batch_index,predicate_index,span_index,:].argmax().item()-1, namespace="srl_labels"))))
            output_dict["srl_predictions"] = predicted_tags
            # Shape: (batch_size, num_predicate_candidates_to_keep, num_spans_to_keep)
            # word_span_mask = torch.gather(word_span_mask, 2, top_span_indices.unsqueeze(1).repeat(1, word_span_mask.shape[1], 1))
            # word_span_mask = torch.gather(word_span_mask, 1, top_predicate_indices.unsqueeze(-1).repeat(1, 1, word_span_mask.shape[-1]))
            if srl_labels is not None:
                original_srl_labels = srl_labels.clone()
                # Shape: (batch_size, num_predicate_candidates_to_keep, num_spans_to_keep)
                srl_labels = torch.gather(srl_labels, 2, top_span_indices.unsqueeze(1).repeat(1, srl_labels.shape[1], 1))
                srl_labels = torch.gather(srl_labels, 1, top_predicate_indices.unsqueeze(-1).repeat(1, 1, srl_labels.shape[-1]))
                if self._lpsmap:
                    # Shape: (batch_size, num_predicate_candidates_to_keep, num_spans_to_keep, num_labels+1)
                    srl_labels = torch.zeros_like(argument_scores).scatter_(dim=-1, index=srl_labels.unsqueeze(-1), src=torch.ones_like(srl_labels.unsqueeze(-1)).to(argument_scores.dtype))
                    loss += LpsmapLoss.apply(argument_scores, argument_probs, srl_labels, combined_mask.unsqueeze(-1).repeat(1, 1, 1, num_labels+1))
                else:
                    loss_entries = torch.nn.CrossEntropyLoss(reduction='none')(argument_scores.view(-1, num_labels+1), srl_labels.view(-1)) # *num_spans_to_keep/num_predicate_candidates_to_keep
                    loss_entries = torch.where(combined_mask.view(-1), loss_entries, torch.zeros_like(loss_entries))
                    loss += loss_entries.sum()/combined_mask.float().sum()
                gold_srl_triples = []
                for i in range(batch_size):
                    gold_srl_triples.append([])
                    for predicate_index, arguments in metadata[i]["srl_frames"]:
                        for span in arguments:
                            gold_srl_triples[-1].append((predicate_index, span))
                self._srl_metric(predicted_tags, gold_srl_triples)

        elif self._predict_srl and srl_seq_predicates is not None:
            # Shape: (batch_size, num_predicates)
            predicate_mask = (srl_seq_predicates > -1).bool()
            # Shape: (batch_size, num_predicates, embedding_size)
            predicate_embeddings = torch.gather(contextualized_embeddings, 1, srl_seq_predicates.unsqueeze(-1).repeat(1, 1, contextualized_embeddings.shape[-1]).clamp(min=0))
            # Shape: (batch_size, num_predicates, sentence_length)
            sequence_mask = (srl_seq_indices > -1).long()
            _, num_predicates, sentence_length = sequence_mask.shape
            # Shape: (batch_size, num_predicates*sentence_length, embedding_size)
            seq_embeddings = torch.gather(contextualized_embeddings, 1, srl_seq_indices.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, contextualized_embeddings.shape[-1]).clamp(min=0))
            # Shape: (batch_size, num_predicates, sentence_length, embedding_size)
            seq_embeddings = seq_embeddings.view(batch_size, num_predicates, sentence_length, seq_embeddings.shape[2])
            # Shape: (batch_size, num_predicates, sentence_length, 2*embedding_size)
            classifier_inputs = torch.cat((seq_embeddings, predicate_embeddings.unsqueeze(2).repeat(1, 1, sentence_length, 1)), dim=-1)
            # Shape: (batch_size, num_predicates, sentence_length, num_labels)
            classifier_outputs = self._srl_scorer(classifier_inputs)
            class_probabilities = torch.softmax(classifier_outputs, dim=-1)
            if not self.training:
                viterbi_output = self.srl_crf.viterbi_tags(
                    classifier_outputs.view(-1, sentence_length, classifier_outputs.shape[-1])[predicate_mask.view(-1),:,:],
                    sequence_mask.view(-1, sentence_length)[predicate_mask.view(-1),:]
                )
                bio_predicted_tags = [
                    [self.vocab.get_token_from_index(tag, namespace="srl_seq_labels") for tag in seq] for (seq, _) in viterbi_output
                ]
            if srl_seq_labels is not None:
                loss += -self.srl_crf(classifier_outputs.view(-1, sentence_length, classifier_outputs.shape[3]), srl_seq_labels.view(-1, sentence_length), sequence_mask.view(-1, sentence_length))/sequence_mask.sum().float()
                if not self.training:
                    from allennlp_models.structured_prediction.models.srl import (
                        convert_bio_tags_to_conll_format,
                    )

                    batch_conll_predicted_tags = [
                        convert_bio_tags_to_conll_format(seq) for seq in bio_predicted_tags
                    ]
                    batch_bio_gold_tags = [
                        seq for example_metadata in metadata for seq in example_metadata["srl_seq_labels"]
                    ]
                    # print(batch_bio_gold_tags)
                    batch_conll_gold_tags = [
                        convert_bio_tags_to_conll_format(tags) for tags in batch_bio_gold_tags
                    ]
                    batch_sentences = [
                        words for example_metadata in metadata for words in example_metadata["srl_seq_words"]
                    ]
                    sentence_offsets = srl_seq_indices[:,:,0]
                    batch_predicate_indices_minus_offset = (srl_seq_predicates-sentence_offsets).view(-1)[predicate_mask.view(-1)].tolist()
                    """for lst1, lst2, bio_gold, sent, pred in zip(batch_conll_predicted_tags, batch_conll_gold_tags, batch_bio_gold_tags, batch_sentences, batch_predicate_indices_minus_offset):
                        print('A', lst1)
                        print('B', lst2)
                        print('C', bio_gold)
                        self._srl_metric(
                            [pred],
                            [sent],
                            [lst1],
                            [lst2]
                        )"""
                    self._srl_metric(
                        batch_predicate_indices_minus_offset,
                        batch_sentences,
                        batch_conll_predicted_tags,
                        batch_conll_gold_tags,
                    )


        if self._predict_ner and not self._ner_sequence:
            span_entity_scores = self._ner_scorer(transformed_span_embeddings).view(-1, self.vocab.get_vocab_size("ner_span_labels"))
            if ner_span_labels is not None:
                ner_span_labels = ner_span_labels.view(-1)
                loss += torch.nn.CrossEntropyLoss()(span_entity_scores, ner_span_labels)
                self._ner_metric(span_entity_scores, ner_span_labels)
                if metadata is not None:
                    output_dict["ner_seq_labels"] = [x["ner_seq_labels"] for x in metadata]
            output_dict["ner_predictions"] = [[self.vocab.get_token_from_index(span_entity_scores[i*span_entity_scores.shape[1]+j,:].argmax().item(), namespace="ner_span_labels") for j in range(num_spans)] for i in range(batch_size)]

        if self._predict_coref:
            if self._non_span_rep_model:
                start_representations = self._start_feedforward2(contextualized_embeddings)
                end_representations = self._end_feedforward2(contextualized_embeddings)
                # Shape: (batch_size, document_length, document_length)
                """antecedent_scores_start_start = self._antecedent_scorer_start_start(
                    start_representations,
                    start_representations
                )
                antecedent_scores_start_end = self._antecedent_scorer_start_end(
                    start_representations,
                    end_representations
                )
                antecedent_scores_end_start = self._antecedent_scorer_end_start(
                    end_representations,
                    start_representations
                )
                antecedent_scores_end_end = self._antecedent_scorer_end_end(
                    end_representations,
                    end_representations
                )"""
                antecedent_scores_start_start = torch.matmul(
                    self._antecedent_scorer_start_start(start_representations),
                    start_representations.permute(0, 2, 1)
                )
                antecedent_scores_start_end = torch.matmul(
                    self._antecedent_scorer_start_end(start_representations),
                    end_representations.permute(0, 2, 1)
                )
                antecedent_scores_end_start = torch.matmul(
                    self._antecedent_scorer_end_start(end_representations),
                    start_representations.permute(0, 2, 1)
                )
                antecedent_scores_end_end = torch.matmul(
                    self._antecedent_scorer_end_end(end_representations),
                    end_representations.permute(0, 2, 1)
                )
                # Shape: (batch_size, num_spans_to_keep, max_antecedents)
                top_antecedent_indices = torch.arange(num_spans_to_keep).to(spans.device).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_spans_to_keep, 1)
                top_antecedent_mask = torch.tril(torch.ones_like(top_antecedent_indices).bool(), diagonal=-1)
                antecedent_mention_scores = top_span_mention_scores.unsqueeze(1).repeat(1, num_spans_to_keep, 1)
                antecedent_starts = top_spans[:,:,0].unsqueeze(1).repeat(1, num_spans_to_keep, 1)
                antecedent_ends = top_spans[:,:,1].unsqueeze(1).repeat(1, num_spans_to_keep, 1)
                max_antecedents = top_antecedent_indices.shape[-1]
                batch_range = torch.arange(batch_size).unsqueeze(-1).repeat(1, num_spans_to_keep*max_antecedents).view(-1)
                # Shape: (batch_size, num_spans_to_keep, max_antecedents)
                antecedent_scores_start_start = antecedent_scores_start_start[
                    batch_range,
                    top_spans[:,:,:1].repeat(1, 1, max_antecedents).view(-1),
                    antecedent_starts.view(-1)
                ].view(batch_size, num_spans_to_keep, max_antecedents)
                antecedent_scores_start_end = antecedent_scores_start_end[
                    batch_range,
                    top_spans[:,:,:1].repeat(1, 1, max_antecedents).view(-1),
                    antecedent_ends.view(-1)
                ].view(batch_size, num_spans_to_keep, max_antecedents)
                antecedent_scores_end_start = antecedent_scores_end_start[
                    batch_range,
                    top_spans[:,:,1:].repeat(1, 1, max_antecedents).view(-1),
                    antecedent_starts.view(-1)
                ].view(batch_size, num_spans_to_keep, max_antecedents)
                antecedent_scores_end_end = antecedent_scores_end_end[
                    batch_range,
                    top_spans[:,:,1:].repeat(1, 1, max_antecedents).view(-1),
                    antecedent_ends.view(-1)
                ].view(batch_size, num_spans_to_keep, max_antecedents)
                # Shape: (batch_size, num_spans_to_keep, max_antecedents)
                coreference_scores = top_span_mention_scores.unsqueeze(-1).repeat(1, 1, max_antecedents)+antecedent_mention_scores+antecedent_scores_start_start+antecedent_scores_start_end+antecedent_scores_end_start+antecedent_scores_end_end
                # top_antecedent_indices = top_antecedent_indices.unsqueeze(0).repeat(batch_size, 1, 1)
                # Shape: (batch_size, num_spans_to_keep)
                span_loss_mask = torch.gather(span_loss_mask, dim=1, index=top_span_indices)
                # Shape: (batch_size, num_spans_to_keep, max_antecedents)
                top_antecedent_mask &= span_loss_mask.unsqueeze(2) & span_loss_mask.unsqueeze(1)
                coreference_scores = util.replace_masked_values(
                    coreference_scores, top_antecedent_mask, util.min_value_of_dtype(coreference_scores.dtype)
                )
                # Shape: (batch_size, num_spans_to_keep, max_antecedents+1)
                coreference_scores = torch.cat((torch.zeros_like(coreference_scores[:,:,:1]), coreference_scores), dim=-1)
                # Shape: (batch_size, num_spans_to_keep, max_antecedents)
                # top_antecedent_mask = top_antecedent_mask.repeat(batch_size, 1, 1)
                flat_top_antecedent_indices = util.flatten_and_batch_shift_indices(
                    top_antecedent_indices, num_spans_to_keep
                )
            else:
                # Compute indices for antecedent spans to consider.
                max_antecedents = min(self._max_antecedents, num_spans_to_keep)

                # Now that we have our variables in terms of num_spans_to_keep, we need to
                # compare span pairs to decide each span's antecedent. Each span can only
                # have prior spans as antecedents, and we only consider up to max_antecedents
                # prior spans. So the first thing we do is construct a matrix mapping a span's
                # index to the indices of its allowed antecedents.

                # Once we have this matrix, we reformat our variables again to get embeddings
                # for all valid antecedents for each span. This gives us variables with shapes
                # like (batch_size, num_spans_to_keep, max_antecedents, embedding_size), which
                # we can use to make coreference decisions between valid span pairs.

                if self._coarse_to_fine:
                    pruned_antecedents = self._coarse_to_fine_pruning(
                        top_span_embeddings, top_span_mention_scores, top_span_mask, max_antecedents
                    )
                else:
                    pruned_antecedents = self._distance_pruning(
                        top_span_embeddings, top_span_mention_scores, max_antecedents
                    )

                # Shape: (batch_size, num_spans_to_keep, max_antecedents) for all 4 tensors
                (
                    top_partial_coreference_scores,
                    top_antecedent_mask,
                    top_antecedent_offsets,
                    top_antecedent_indices,
                ) = pruned_antecedents

                flat_top_antecedent_indices = util.flatten_and_batch_shift_indices(
                    top_antecedent_indices, num_spans_to_keep
                )

                # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
                top_antecedent_embeddings = util.batched_index_select(
                    top_span_embeddings, top_antecedent_indices, flat_top_antecedent_indices
                )
                # Shape: (batch_size, num_spans_to_keep, 1 + max_antecedents)
                coreference_scores = self._compute_coreference_scores(
                    top_span_embeddings,
                    top_antecedent_embeddings,
                    top_partial_coreference_scores,
                    top_antecedent_mask,
                    top_antecedent_offsets,
                )

                for _ in range(self._inference_order - 1):
                    dummy_mask = top_antecedent_mask.new_ones(batch_size, num_spans_to_keep, 1)
                    # Shape: (batch_size, num_spans_to_keep, 1 + max_antecedents,)
                    top_antecedent_with_dummy_mask = torch.cat([dummy_mask, top_antecedent_mask], -1)
                    # Shape: (batch_size, num_spans_to_keep, 1 + max_antecedents)
                    attention_weight = util.masked_softmax(
                        coreference_scores, top_antecedent_with_dummy_mask, memory_efficient=True
                    )
                    # Shape: (batch_size, num_spans_to_keep, 1 + max_antecedents, embedding_size)
                    top_antecedent_with_dummy_embeddings = torch.cat(
                        [top_span_embeddings.unsqueeze(2), top_antecedent_embeddings], 2
                    )
                    # Shape: (batch_size, num_spans_to_keep, embedding_size)
                    attended_embeddings = util.weighted_sum(
                        top_antecedent_with_dummy_embeddings, attention_weight
                    )
                    # Shape: (batch_size, num_spans_to_keep, embedding_size)
                    top_span_embeddings = self._span_updating_gated_sum(
                        top_span_embeddings, attended_embeddings
                    )

                    # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
                    top_antecedent_embeddings = util.batched_index_select(
                        top_span_embeddings, top_antecedent_indices, flat_top_antecedent_indices
                    )
                    # Shape: (batch_size, num_spans_to_keep, 1 + max_antecedents)
                    coreference_scores = self._compute_coreference_scores(
                        top_span_embeddings,
                        top_antecedent_embeddings,
                        top_partial_coreference_scores,
                        top_antecedent_mask,
                        top_antecedent_offsets,
                    )

            # We now have, for each span which survived the pruning stage,
            # a predicted antecedent. This implies a clustering if we group
            # mentions which refer to each other in a chain.
            # Shape: (batch_size, num_spans_to_keep)
            _, predicted_antecedents = coreference_scores.max(2)
            # Subtract one here because index 0 is the "no antecedent" class,
            # so this makes the indices line up with actual spans if the prediction
            # is greater than -1.
            predicted_antecedents -= 1


            output_dict["antecedent_indices"] = top_antecedent_indices
            output_dict["antecedent_mask"] = top_antecedent_mask
            output_dict["predicted_antecedents"] = predicted_antecedents
            if self._bug_fix:
                coreference_log_probs = util.masked_log_softmax(
                    coreference_scores, top_antecedent_mask
                )
            else:
                coreference_log_probs = util.masked_log_softmax(
                    coreference_scores, top_span_mask.unsqueeze(-1)
                )
            output_dict["coreference_log_probs"] = coreference_log_probs
            output_dict["coreference_scores"] = coreference_scores
            if span_labels is not None:
                # Find the gold labels for the spans which we kept.
                # Shape: (batch_size, num_spans_to_keep, 1)
                pruned_gold_labels = util.batched_index_select(
                    span_labels.unsqueeze(-1), top_span_indices, flat_top_span_indices
                )

                # Shape: (batch_size, num_spans_to_keep, max_antecedents)
                antecedent_labels = util.batched_index_select(
                    pruned_gold_labels, top_antecedent_indices, flat_top_antecedent_indices
                ).squeeze(-1)
                antecedent_labels = util.replace_masked_values(
                    antecedent_labels, top_antecedent_mask, -100
                )

                # Compute labels.
                # Shape: (batch_size, num_spans_to_keep, max_antecedents + 1)
                gold_antecedent_labels = self._compute_antecedent_gold_labels(
                    pruned_gold_labels, antecedent_labels
                )
                # Now, compute the loss using the negative marginal log-likelihood.
                # This is equal to the log of the sum of the probabilities of all antecedent predictions
                # that would be consistent with the data, in the sense that we are minimising, for a
                # given span, the negative marginal log likelihood of all antecedents which are in the
                # same gold cluster as the span we are currently considering. Each span i predicts a
                # single antecedent j, but there might be several prior mentions k in the same
                # coreference cluster that would be valid antecedents. Our loss is the sum of the
                # probability assigned to all valid antecedents. This is a valid objective for
                # clustering as we don't mind which antecedent is predicted, so long as they are in
                #  the same coreference cluster.

                correct_antecedent_log_probs = coreference_log_probs + gold_antecedent_labels.log()
                negative_marginal_log_likelihood = -util.logsumexp(correct_antecedent_log_probs).mean()

                self._mention_recall(top_spans, metadata)
                self._conll_coref_scores(
                    top_spans, top_antecedent_indices, metadata, predicted_antecedents
                )
                loss += negative_marginal_log_likelihood
                output_dict["gold_clusters"] = [x["clusters"] for x in metadata]

        output_dict["loss"] = loss

        if metadata is not None:
            output_dict["document"] = [x["original_text"] for x in metadata]
            output_dict["document_id"] = [x["document_id"] for x in metadata]
        return output_dict

    def get_ner_transition_scores(self, include_constraints: bool) -> torch.Tensor:
        num_tags = self.vocab.get_vocab_size("ner_seq_labels")
        # Augment transitions matrix with start and end transitions
        start_tag = num_tags
        end_tag = num_tags + 1
        transitions = torch.Tensor(num_tags + 2, num_tags + 2).fill_(-10000.0).to(self.ner_crf.transitions.device)
        if include_constraints:
            constraint_mask = self.ner_crf._constraint_mask.permute(1, 0)
        else:
            constraint_mask = torch.ones((num_tags+2, num_tags+2), dtype=torch.float32).to(transitions.device)

        # Apply transition constraints
        constrained_transitions = self.ner_crf.transitions * constraint_mask[
            :num_tags+2, :num_tags+2
        ] + -10000.0 * (1 - constraint_mask[:num_tags+2, :num_tags+2])
        transitions[:num_tags+2, :num_tags+2] = constrained_transitions.data

        if self.ner_crf.include_start_end_transitions:
            transitions[
                start_tag, :num_tags+2
            ] = transitions[start_tag, :num_tags+2] * constraint_mask[
                start_tag, :num_tags+2
            ].data + -10000.0 * (
                1 - constraint_mask[start_tag, :num_tags+2].detach()
            )
            transitions[:num_tags+2, end_tag] = transitions[:num_tags+2, end_tag] * constraint_mask[
                :num_tags+2, end_tag
            ].data + -10000.0 * (1 - constraint_mask[:num_tags+2, end_tag].detach())
        else:
            transitions[start_tag, :num_tags] = -10000.0 * (
                1 - constraint_mask[start_tag, :num_tags].detach()
            )
            transitions[:num_tags, end_tag] = -10000.0 * (
                1 - constraint_mask[:num_tags, end_tag].detach()
            )

        return transitions

    def srl_seq_make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Does constrained viterbi decoding on class probabilities output in :func:`forward`.  The
        constraint simply specifies that the output tags must be a valid BIO sequence.  We add a
        `"tags"` key to the dictionary with the result.

        NOTE: First, we decode a BIO sequence on top of the wordpieces. This is important; viterbi
        decoding produces low quality output if you decode on top of word representations directly,
        because the model gets confused by the 'missing' positions (which is sensible as it is trained
        to perform tagging on wordpieces, not words).

        Secondly, it's important that the indices we use to recover words from the wordpieces are the
        start_offsets (i.e offsets which correspond to using the first wordpiece of words which are
        tokenized into multiple wordpieces) as otherwise, we might get an ill-formed BIO sequence
        when we select out the word tags from the wordpiece tags. This happens in the case that a word
        is split into multiple word pieces, and then we take the last tag of the word, which might
        correspond to, e.g, I-V, which would not be allowed as it is not preceeded by a B tag.
        """
        all_predictions = output_dict["class_probabilities"]
        sequence_lengths = get_lengths_from_binary_sequence_mask(output_dict["mask"]).data.tolist()

        if all_predictions.dim() == 3:
            predictions_list = [
                all_predictions[i].detach().cpu() for i in range(all_predictions.size(0))
            ]
        else:
            predictions_list = [all_predictions]
        all_tags = []
        transition_matrix = self.get_viterbi_pairwise_potentials()
        start_transitions = self.get_start_transitions()
        # **************** Different ********************
        # We add in the offsets here so we can compute the un-wordpieced tags.
        for predictions, length in zip(
            predictions_list, sequence_lengths
        ):
            max_likelihood_sequence, _ = viterbi_decode(
                predictions[:length], transition_matrix, allowed_start_transitions=start_transitions
            )
            tags = [
                self.vocab.get_token_from_index(x, namespace=self._label_namespace)
                for x in max_likelihood_sequence
            ]

            all_tags.append(tags)
            # print(all_tags)
        output_dict["tags"] = all_tags
        return output_dict

    @overrides
    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]):
        """
        Converts the list of spans and predicted antecedent indices into clusters
        of spans for each element in the batch.

        # Parameters

        output_dict : `Dict[str, torch.Tensor]`, required.
            The result of calling :func:`forward` on an instance or batch of instances.

        # Returns

        The same output dictionary, but with an additional `clusters` key:

        clusters : `List[List[List[Tuple[int, int]]]]`
            A nested list, representing, for each instance in the batch, the list of clusters,
            which are in turn comprised of a list of (start, end) inclusive spans into the
            original document.
        """

        # A tensor of shape (batch_size, num_spans_to_keep, 2), representing
        # the start and end indices of each span.
        batch_top_spans = output_dict["top_spans"].detach().cpu()

        # A tensor of shape (batch_size, num_spans_to_keep) representing, for each span,
        # the index into `antecedent_indices` which specifies the antecedent span. Additionally,
        # the index can be -1, specifying that the span has no predicted antecedent.
        batch_predicted_antecedents = output_dict["predicted_antecedents"].detach().cpu()

        # A tensor of shape (num_spans_to_keep, max_antecedents), representing the indices
        # of the predicted antecedents with respect to the 2nd dimension of `batch_top_spans`
        # for each antecedent we considered.
        batch_antecedent_indices = output_dict["antecedent_indices"].detach().cpu()
        batch_clusters: List[List[List[Tuple[int, int]]]] = []

        # Calling zip() on two tensors results in an iterator over their
        # first dimension. This is iterating over instances in the batch.
        for top_spans, predicted_antecedents, antecedent_indices in zip(
            batch_top_spans, batch_predicted_antecedents, batch_antecedent_indices
        ):
            spans_to_cluster_ids: Dict[Tuple[int, int], int] = {}
            clusters: List[List[Tuple[int, int]]] = []

            for i, (span, predicted_antecedent) in enumerate(zip(top_spans, predicted_antecedents)):
                if predicted_antecedent < 0:
                    # We don't care about spans which are
                    # not co-referent with anything.
                    continue

                # Find the right cluster to update with this span.
                # To do this, we find the row in `antecedent_indices`
                # corresponding to this span we are considering.
                # The predicted antecedent is then an index into this list
                # of indices, denoting the span from `top_spans` which is the
                # most likely antecedent.
                predicted_index = antecedent_indices[i, predicted_antecedent]

                antecedent_span = (
                    top_spans[predicted_index, 0].item(),
                    top_spans[predicted_index, 1].item(),
                )

                # Check if we've seen the span before.
                if antecedent_span in spans_to_cluster_ids:
                    predicted_cluster_id: int = spans_to_cluster_ids[antecedent_span]
                else:
                    # We start a new cluster.
                    predicted_cluster_id = len(clusters)
                    # Append a new cluster containing only this span.
                    clusters.append([antecedent_span])
                    # Record the new id of this span.
                    spans_to_cluster_ids[antecedent_span] = predicted_cluster_id

                # Now add the span we are currently considering.
                span_start, span_end = span[0].item(), span[1].item()
                clusters[predicted_cluster_id].append((span_start, span_end))
                spans_to_cluster_ids[(span_start, span_end)] = predicted_cluster_id
            batch_clusters.append(clusters)

        output_dict["clusters"] = batch_clusters
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        mention_recall = self._mention_recall.get_metric(reset)
        coref_precision, coref_recall, coref_f1 = self._conll_coref_scores.get_metric(reset)

        metrics = {
            "coref_precision": coref_precision,
            "coref_recall": coref_recall,
            "coref_f1": coref_f1,
            "mention_recall": mention_recall,
        }
        if self._predict_srl:
            f1 = self._srl_metric.get_metric(reset)
            if isinstance(f1, dict):
                metrics["srl_precision"] = f1["precision-overall"]
                metrics["srl_recall"] = f1["recall-overall"]
                f1 = f1["f1-measure-overall"]
            metrics["srl_f1"] = f1
        if self._predict_ner:
            if self._ner_sequence:
                ner_metrics = self._ner_metric.get_metric(reset)
                metrics["ner_f1"] = ner_metrics["f1-measure-overall"]
                metrics["ner_precision"] = ner_metrics["precision-overall"]
                metrics["ner_recall"] = ner_metrics["recall-overall"]
            else:
                try:
                    ner_metrics = self._ner_metric.get_metric(reset)
                    metrics["ner_f1"] = ner_metrics["fscore"]
                    metrics["precision"] = ner_metrics["precision"]
                    metrics["recall"] = ner_metrics["recall"]
                except RuntimeError:
                    metrics["ner_f1"] = 0
        return metrics

    @staticmethod
    def _generate_valid_antecedents(
        num_spans_to_keep: int, max_antecedents: int, device: int
    ) -> Tuple[torch.IntTensor, torch.IntTensor, torch.BoolTensor]:
        """
        This method generates possible antecedents per span which survived the pruning
        stage. This procedure is `generic across the batch`. The reason this is the case is
        that each span in a batch can be coreferent with any previous span, but here we
        are computing the possible `indices` of these spans. So, regardless of the batch,
        the 1st span _cannot_ have any antecedents, because there are none to select from.
        Similarly, each element can only predict previous spans, so this returns a matrix
        of shape (num_spans_to_keep, max_antecedents), where the (i,j)-th index is equal to
        (i - 1) - j if j <= i, or zero otherwise.

        # Parameters

        num_spans_to_keep : `int`, required.
            The number of spans that were kept while pruning.
        max_antecedents : `int`, required.
            The maximum number of antecedent spans to consider for every span.
        device : `int`, required.
            The CUDA device to use.

        # Returns

        valid_antecedent_indices : `torch.LongTensor`
            The indices of every antecedent to consider with respect to the top k spans.
            Has shape `(num_spans_to_keep, max_antecedents)`.
        valid_antecedent_offsets : `torch.LongTensor`
            The distance between the span and each of its antecedents in terms of the number
            of considered spans (i.e not the word distance between the spans).
            Has shape `(1, max_antecedents)`.
        valid_antecedent_mask : `torch.BoolTensor`
            The mask representing whether each antecedent span is valid. Required since
            different spans have different numbers of valid antecedents. For example, the first
            span in the document should have no valid antecedents.
            Has shape `(1, num_spans_to_keep, max_antecedents)`.
        """
        # Shape: (num_spans_to_keep, 1)
        target_indices = util.get_range_vector(num_spans_to_keep, device).unsqueeze(1)

        # Shape: (1, max_antecedents)
        valid_antecedent_offsets = (util.get_range_vector(max_antecedents, device) + 1).unsqueeze(0)

        # This is a broadcasted subtraction.
        # Shape: (num_spans_to_keep, max_antecedents)
        raw_antecedent_indices = target_indices - valid_antecedent_offsets

        # In our matrix of indices, the upper triangular part will be negative
        # because the offsets will be > the target indices. We want to mask these,
        # because these are exactly the indices which we don't want to predict, per span.
        # Shape: (1, num_spans_to_keep, max_antecedents)
        valid_antecedent_mask = (raw_antecedent_indices >= 0).unsqueeze(0)

        # Shape: (num_spans_to_keep, max_antecedents)
        valid_antecedent_indices = F.relu(raw_antecedent_indices.float()).long()
        return valid_antecedent_indices, valid_antecedent_offsets, valid_antecedent_mask

    def _distance_pruning(
        self,
        top_span_embeddings: torch.FloatTensor,
        top_span_mention_scores: torch.FloatTensor,
        max_antecedents: int,
    ) -> Tuple[torch.FloatTensor, torch.BoolTensor, torch.LongTensor, torch.LongTensor]:
        """
        Generates antecedents for each span and prunes down to `max_antecedents`. This method
        prunes antecedents only based on distance (i.e. number of intervening spans). The closest
        antecedents are kept.

        # Parameters

        top_span_embeddings: `torch.FloatTensor`, required.
            The embeddings of the top spans.
            (batch_size, num_spans_to_keep, embedding_size).
        top_span_mention_scores: `torch.FloatTensor`, required.
            The mention scores of the top spans.
            (batch_size, num_spans_to_keep).
        max_antecedents: `int`, required.
            The maximum number of antecedents to keep for each span.

        # Returns

        top_partial_coreference_scores: `torch.FloatTensor`
            The partial antecedent scores for each span-antecedent pair. Computed by summing
            the span mentions scores of the span and the antecedent. This score is partial because
            compared to the full coreference scores, it lacks the interaction term
            w * FFNN([g_i, g_j, g_i * g_j, features]).
            (batch_size, num_spans_to_keep, max_antecedents)
        top_antecedent_mask: `torch.BoolTensor`
            The mask representing whether each antecedent span is valid. Required since
            different spans have different numbers of valid antecedents. For example, the first
            span in the document should have no valid antecedents.
            (batch_size, num_spans_to_keep, max_antecedents)
        top_antecedent_offsets: `torch.LongTensor`
            The distance between the span and each of its antecedents in terms of the number
            of considered spans (i.e not the word distance between the spans).
            (batch_size, num_spans_to_keep, max_antecedents)
        top_antecedent_indices: `torch.LongTensor`
            The indices of every antecedent to consider with respect to the top k spans.
            (batch_size, num_spans_to_keep, max_antecedents)
        """
        # These antecedent matrices are independent of the batch dimension - they're just a function
        # of the span's position in top_spans.
        # The spans are in document order, so we can just use the relative
        # index of the spans to know which other spans are allowed antecedents.

        num_spans_to_keep = top_span_embeddings.size(1)
        device = util.get_device_of(top_span_embeddings)

        # Shapes:
        # (num_spans_to_keep, max_antecedents),
        # (1, max_antecedents),
        # (1, num_spans_to_keep, max_antecedents)
        (
            top_antecedent_indices,
            top_antecedent_offsets,
            top_antecedent_mask,
        ) = self._generate_valid_antecedents(  # noqa
            num_spans_to_keep, max_antecedents, device
        )

        # Shape: (batch_size, num_spans_to_keep, max_antecedents)
        top_antecedent_mention_scores = util.flattened_index_select(
            top_span_mention_scores.unsqueeze(-1), top_antecedent_indices
        ).squeeze(-1)

        # Shape: (batch_size, num_spans_to_keep, max_antecedents) * 4
        top_partial_coreference_scores = (
            top_span_mention_scores.unsqueeze(-1) + top_antecedent_mention_scores
        )
        top_antecedent_indices = top_antecedent_indices.unsqueeze(0).expand_as(
            top_partial_coreference_scores
        )
        top_antecedent_offsets = top_antecedent_offsets.unsqueeze(0).expand_as(
            top_partial_coreference_scores
        )
        top_antecedent_mask = top_antecedent_mask.expand_as(top_partial_coreference_scores)

        return (
            top_partial_coreference_scores,
            top_antecedent_mask,
            top_antecedent_offsets,
            top_antecedent_indices,
        )

    def _coarse_to_fine_pruning(
        self,
        top_span_embeddings: torch.FloatTensor,
        top_span_mention_scores: torch.FloatTensor,
        top_span_mask: torch.BoolTensor,
        max_antecedents: int,
    ) -> Tuple[torch.FloatTensor, torch.BoolTensor, torch.LongTensor, torch.LongTensor]:
        """
        Generates antecedents for each span and prunes down to `max_antecedents`. This method
        prunes antecedents using a fast bilinar interaction score between a span and a candidate
        antecedent, and the highest-scoring antecedents are kept.

        # Parameters

        top_span_embeddings: `torch.FloatTensor`, required.
            The embeddings of the top spans.
            (batch_size, num_spans_to_keep, embedding_size).
        top_span_mention_scores: `torch.FloatTensor`, required.
            The mention scores of the top spans.
            (batch_size, num_spans_to_keep).
        top_span_mask: `torch.BoolTensor`, required.
            The mask for the top spans.
            (batch_size, num_spans_to_keep).
        max_antecedents: `int`, required.
            The maximum number of antecedents to keep for each span.

        # Returns

        top_partial_coreference_scores: `torch.FloatTensor`
            The partial antecedent scores for each span-antecedent pair. Computed by summing
            the span mentions scores of the span and the antecedent as well as a bilinear
            interaction term. This score is partial because compared to the full coreference scores,
            it lacks the interaction term
            `w * FFNN([g_i, g_j, g_i * g_j, features])`.
            `(batch_size, num_spans_to_keep, max_antecedents)`
        top_antecedent_mask: `torch.BoolTensor`
            The mask representing whether each antecedent span is valid. Required since
            different spans have different numbers of valid antecedents. For example, the first
            span in the document should have no valid antecedents.
            `(batch_size, num_spans_to_keep, max_antecedents)`
        top_antecedent_offsets: `torch.LongTensor`
            The distance between the span and each of its antecedents in terms of the number
            of considered spans (i.e not the word distance between the spans).
            `(batch_size, num_spans_to_keep, max_antecedents)`
        top_antecedent_indices: `torch.LongTensor`
            The indices of every antecedent to consider with respect to the top k spans.
            `(batch_size, num_spans_to_keep, max_antecedents)`
        """
        batch_size, num_spans_to_keep = top_span_embeddings.size()[:2]
        device = util.get_device_of(top_span_embeddings)

        # Shape: (1, num_spans_to_keep, num_spans_to_keep)
        _, _, valid_antecedent_mask = self._generate_valid_antecedents(
            num_spans_to_keep, num_spans_to_keep, device
        )

        mention_one_score = top_span_mention_scores.unsqueeze(1)
        mention_two_score = top_span_mention_scores.unsqueeze(2)
        bilinear_weights = self._coarse2fine_scorer(top_span_embeddings).transpose(1, 2)
        bilinear_score = torch.matmul(top_span_embeddings, bilinear_weights)
        # Shape: (batch_size, num_spans_to_keep, num_spans_to_keep); broadcast op
        partial_antecedent_scores = mention_one_score + mention_two_score + bilinear_score

        # Shape: (batch_size, num_spans_to_keep, num_spans_to_keep); broadcast op
        span_pair_mask = top_span_mask.unsqueeze(-1) & valid_antecedent_mask

        # Shape:
        # (batch_size, num_spans_to_keep, max_antecedents) * 3
        (
            top_partial_coreference_scores,
            top_antecedent_mask,
            top_antecedent_indices,
        ) = util.masked_topk(partial_antecedent_scores, span_pair_mask, max_antecedents)

        top_span_range = util.get_range_vector(num_spans_to_keep, device)
        # Shape: (num_spans_to_keep, num_spans_to_keep); broadcast op
        valid_antecedent_offsets = top_span_range.unsqueeze(-1) - top_span_range.unsqueeze(0)

        # TODO: we need to make `batched_index_select` more general to make this less awkward.
        top_antecedent_offsets = util.batched_index_select(
            valid_antecedent_offsets.unsqueeze(0)
            .expand(batch_size, num_spans_to_keep, num_spans_to_keep)
            .reshape(batch_size * num_spans_to_keep, num_spans_to_keep, 1),
            top_antecedent_indices.view(-1, max_antecedents),
        ).reshape(batch_size, num_spans_to_keep, max_antecedents)

        return (
            top_partial_coreference_scores,
            top_antecedent_mask,
            top_antecedent_offsets,
            top_antecedent_indices,
        )

    def _compute_span_pair_embeddings(
        self,
        top_span_embeddings: torch.FloatTensor,
        antecedent_embeddings: torch.FloatTensor,
        antecedent_offsets: torch.FloatTensor,
    ):
        """
        Computes an embedding representation of pairs of spans for the pairwise scoring function
        to consider. This includes both the original span representations, the element-wise
        similarity of the span representations, and an embedding representation of the distance
        between the two spans.

        # Parameters

        top_span_embeddings : `torch.FloatTensor`, required.
            Embedding representations of the top spans. Has shape
            (batch_size, num_spans_to_keep, embedding_size).
        antecedent_embeddings : `torch.FloatTensor`, required.
            Embedding representations of the antecedent spans we are considering
            for each top span. Has shape
            (batch_size, num_spans_to_keep, max_antecedents, embedding_size).
        antecedent_offsets : `torch.IntTensor`, required.
            The offsets between each top span and its antecedent spans in terms
            of spans we are considering. Has shape (batch_size, num_spans_to_keep, max_antecedents).

        # Returns

        span_pair_embeddings : `torch.FloatTensor`
            Embedding representation of the pair of spans to consider. Has shape
            (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        """
        # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        target_embeddings = top_span_embeddings.unsqueeze(2).expand_as(antecedent_embeddings)

        # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        antecedent_distance_embeddings = self._distance_embedding(
            util.bucket_values(antecedent_offsets, num_total_buckets=self._num_distance_buckets)
        )

        # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        span_pair_embeddings = torch.cat(
            [
                target_embeddings,
                antecedent_embeddings,
                antecedent_embeddings * target_embeddings,
                antecedent_distance_embeddings,
            ],
            -1,
        )
        return span_pair_embeddings

    @staticmethod
    def _compute_antecedent_gold_labels(
        top_span_labels: torch.IntTensor, antecedent_labels: torch.IntTensor
    ):
        """
        Generates a binary indicator for every pair of spans. This label is one if and
        only if the pair of spans belong to the same cluster. The labels are augmented
        with a dummy antecedent at the zeroth position, which represents the prediction
        that a span does not have any antecedent.

        # Parameters

        top_span_labels : `torch.IntTensor`, required.
            The cluster id label for every span. The id is arbitrary,
            as we just care about the clustering. Has shape (batch_size, num_spans_to_keep).
        antecedent_labels : `torch.IntTensor`, required.
            The cluster id label for every antecedent span. The id is arbitrary,
            as we just care about the clustering. Has shape
            (batch_size, num_spans_to_keep, max_antecedents).

        # Returns

        pairwise_labels_with_dummy_label : `torch.FloatTensor`
            A binary tensor representing whether a given pair of spans belong to
            the same cluster in the gold clustering.
            Has shape (batch_size, num_spans_to_keep, max_antecedents + 1).

        """
        # Shape: (batch_size, num_spans_to_keep, max_antecedents)
        target_labels = top_span_labels.expand_as(antecedent_labels)
        same_cluster_indicator = (target_labels == antecedent_labels).float()
        non_dummy_indicator = (target_labels >= 0).float()
        pairwise_labels = same_cluster_indicator * non_dummy_indicator

        # Shape: (batch_size, num_spans_to_keep, 1)
        dummy_labels = (1 - pairwise_labels).prod(-1, keepdim=True)

        # Shape: (batch_size, num_spans_to_keep, max_antecedents + 1)
        pairwise_labels_with_dummy_label = torch.cat([dummy_labels, pairwise_labels], -1)
        return pairwise_labels_with_dummy_label

    def _compute_coreference_scores(
        self,
        top_span_embeddings: torch.FloatTensor,
        top_antecedent_embeddings: torch.FloatTensor,
        top_partial_coreference_scores: torch.FloatTensor,
        top_antecedent_mask: torch.BoolTensor,
        top_antecedent_offsets: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Computes scores for every pair of spans. Additionally, a dummy label is included,
        representing the decision that the span is not coreferent with anything. For the dummy
        label, the score is always zero. For the true antecedent spans, the score consists of
        the pairwise antecedent score and the unary mention scores for the span and its
        antecedent. The factoring allows the model to blame many of the absent links on bad
        spans, enabling the pruning strategy used in the forward pass.

        # Parameters

        top_span_embeddings : `torch.FloatTensor`, required.
            Embedding representations of the kept spans. Has shape
            (batch_size, num_spans_to_keep, embedding_size)
        top_antecedent_embeddings: `torch.FloatTensor`, required.
            The embeddings of antecedents for each span candidate. Has shape
            (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        top_partial_coreference_scores : `torch.FloatTensor`, required.
            Sum of span mention score and antecedent mention score. The coarse to fine settings
            has an additional term which is the coarse bilinear score.
            (batch_size, num_spans_to_keep, max_antecedents).
        top_antecedent_mask : `torch.BoolTensor`, required.
            The mask for valid antecedents.
            (batch_size, num_spans_to_keep, max_antecedents).
        top_antecedent_offsets : `torch.FloatTensor`, required.
            The distance between the span and each of its antecedents in terms of the number
            of considered spans (i.e not the word distance between the spans).
            (batch_size, num_spans_to_keep, max_antecedents).

        # Returns

        coreference_scores : `torch.FloatTensor`
            A tensor of shape (batch_size, num_spans_to_keep, max_antecedents + 1),
            representing the unormalised score for each (span, antecedent) pair
            we considered.

        """
        # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        span_pair_embeddings = self._compute_span_pair_embeddings(
            top_span_embeddings, top_antecedent_embeddings, top_antecedent_offsets
        )

        # Shape: (batch_size, num_spans_to_keep, max_antecedents)
        antecedent_scores = self._antecedent_scorer(
            self._antecedent_feedforward(span_pair_embeddings)
        ).squeeze(-1)
        antecedent_scores += top_partial_coreference_scores
        antecedent_scores = util.replace_masked_values(
            antecedent_scores, top_antecedent_mask, util.min_value_of_dtype(antecedent_scores.dtype)
        )

        # Shape: (batch_size, num_spans_to_keep, 1)
        shape = [antecedent_scores.size(0), antecedent_scores.size(1), 1]
        dummy_scores = antecedent_scores.new_zeros(*shape)

        # Shape: (batch_size, num_spans_to_keep, max_antecedents + 1)
        coreference_scores = torch.cat([dummy_scores, antecedent_scores], -1)
        return coreference_scores

    default_predictor = "coreference_resolution"
