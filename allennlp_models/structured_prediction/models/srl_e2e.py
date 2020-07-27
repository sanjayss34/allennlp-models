from typing import Dict, List, Any, Union
import math
import time

from overrides import overrides
import torch
from torch.nn.modules import Linear, Dropout
import torch.nn.functional as F
from transformers.modeling_bert import BertModel

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, GatedSum
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor, EndpointSpanExtractor
from allennlp.nn import InitializerApplicator, util
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, viterbi_decode

from allennlp_models.structured_prediction.metrics.srl_eval_scorer import (
    DEFAULT_SRL_EVAL_PATH,
    SrlEvalScorer,
)


@Model.register("srl_e2e")
class SrlE2e(Model):
    """

    # Parameters

    vocab : `Vocabulary`, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    model : `Union[str, BertModel]`, required.
        A string describing the BERT model to load or an already constructed BertModel.
    initializer : `InitializerApplicator`, optional (default=`InitializerApplicator()`)
        Used to initialize the model parameters.
    label_smoothing : `float`, optional (default = `0.0`)
        Whether or not to use label smoothing on the labels when computing cross entropy loss.
    ignore_span_metric : `bool`, optional (default = `False`)
        Whether to calculate span loss, which is irrelevant when predicting BIO for Open Information Extraction.
    srl_eval_path : `str`, optional (default=`DEFAULT_SRL_EVAL_PATH`)
        The path to the srl-eval.pl script. By default, will use the srl-eval.pl included with allennlp,
        which is located at allennlp/tools/srl-eval.pl . If `None`, srl-eval.pl is not used.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        bert_model: Union[str, BertModel],
        mention_feedforward: FeedForward,
        context_layer: Seq2SeqEncoder = None,
        embedding_dropout: float = 0.0,
        initializer: InitializerApplicator = InitializerApplicator(),
        max_span_width: int = 30,
        feature_size: int = 10,
        spans_per_word: float = 100,
        label_smoothing: float = None,
        ignore_span_metric: bool = False,
        srl_eval_path: str = DEFAULT_SRL_EVAL_PATH,
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)

        if isinstance(bert_model, str):
            self.bert_model = BertModel.from_pretrained(bert_model)
        else:
            self.bert_model = bert_model

        self.num_classes = self.vocab.get_vocab_size("span_labels")
        if srl_eval_path is not None:
            # For the span based evaluation, we don't want to consider labels
            # for verb, because the verb index is provided to the model.
            self.span_metric = SrlEvalScorer(srl_eval_path, ignore_classes=["V"])
        else:
            self.span_metric = None
        self.tag_projection_layer = Linear(self.bert_model.config.hidden_size, self.num_classes)

        self.embedding_dropout = Dropout(p=embedding_dropout)
        self._label_smoothing = label_smoothing
        self.ignore_span_metric = ignore_span_metric

        self._mention_feedforward = TimeDistributed(mention_feedforward)
        self._mention_scorer = TimeDistributed(
            torch.nn.Linear(mention_feedforward.get_output_dim(), 1)
        )

        self._attentive_span_extractor = SelfAttentiveSpanExtractor(
            input_dim=self.bert_model.config.hidden_size
        )
        self.span_representation_dim = self._attentive_span_extractor.get_output_dim()
        self._context_layer = context_layer
        if context_layer is not None:
            self._endpoint_span_extractor = EndpointSpanExtractor(
                context_layer.get_output_dim(),
                combination="x,y",
                num_width_embeddings=max_span_width,
                span_width_embedding_dim=feature_size,
                bucket_widths=False,
            )
            self.span_representation_dim = self._endpoint_span_extractor.get_output_dim()

        self.hidden_layer = torch.nn.Sequential(
                                torch.nn.Linear(self.span_representation_dim+self.bert_model.config.hidden_size, self.span_representation_dim, bias=False),
                                torch.nn.ReLU()
                            )
        self.output_layer = torch.nn.Linear(self.span_representation_dim, self.num_classes-1, bias=False)

        self._max_span_width = max_span_width
        self._spans_per_word = spans_per_word
        self._ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self._bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        initializer(self)

    def forward(  # type: ignore
        self,
        tokens: TextFieldTensors,
        verb_indicator: torch.Tensor,
        sentence_end: torch.LongTensor,
        spans: torch.LongTensor,
        span_labels: torch.LongTensor,
        metadata: List[Any],
        tags: torch.LongTensor = None,
    ):

        """
        # Parameters

        tokens : `TextFieldTensors`, required
            The output of `TextField.as_array()`, which should typically be passed directly to a
            `TextFieldEmbedder`. For this model, this must be a `SingleIdTokenIndexer` which
            indexes wordpieces from the BERT vocabulary.
        verb_indicator: `torch.LongTensor`, required.
            An integer `SequenceFeatureField` representation of the position of the verb
            in the sentence. This should have shape (batch_size, num_tokens) and importantly, can be
            all zeros, in the case that the sentence has no verbal predicate.
        tags : `torch.LongTensor`, optional (default = `None`)
            A torch tensor representing the sequence of integer gold class labels
            of shape `(batch_size, num_tokens)`
        metadata : `List[Dict[str, Any]]`, optional, (default = `None`)
            metadata containg the original words in the sentence, the verb to compute the
            frame for, and start offsets for converting wordpieces back to a sequence of words,
            under 'words', 'verb' and 'offsets' keys, respectively.

        # Returns

        An output dictionary consisting of:
        logits : `torch.FloatTensor`
            A tensor of shape `(batch_size, num_tokens, tag_vocab_size)` representing
            unnormalised log probabilities of the tag classes.
        class_probabilities : `torch.FloatTensor`
            A tensor of shape `(batch_size, num_tokens, tag_vocab_size)` representing
            a distribution of the tag classes per word.
        loss : `torch.FloatTensor`, optional
            A scalar loss to be optimised.
        """
        mask = get_text_field_mask(tokens)
        start = time.time()
        bert_embeddings, _ = self.bert_model(
            input_ids=util.get_token_ids_from_text_field_tensors(tokens),
            # token_type_ids=verb_indicator,
            attention_mask=mask,
        )

        # Shape: (batch_size, num_spans)
        span_mask = (spans[:, :, 0] >= 0).squeeze(-1)
        # SpanFields return -1 when they are used as padding. As we do
        # some comparisons based on span widths when we attend over the
        # span representations that we generate from these indices, we
        # need them to be <= 0. This is only relevant in edge cases where
        # the number of spans we consider after the pruning stage is >= the
        # total number of spans, because in this case, it is possible we might
        # consider a masked span.
        # Shape: (batch_size, num_spans, 2)
        spans = F.relu(spans.float()).long()

        embedded_text_input = self.embedding_dropout(bert_embeddings)
        batch_size, sequence_length, _ = embedded_text_input.size()
        # Shape: (batch_size, num_spans, emebedding_size)
        attended_span_embeddings = self._attentive_span_extractor(bert_embeddings, spans)

        if self._context_layer is not None:
            contextualized_embeddings = self._context_layer(embedded_text_input, mask)
            # Shape: (batch_size, num_spans, 2 * encoding_dim + feature_size)
            endpoint_span_embeddings = self._endpoint_span_extractor(contextualized_embeddings, spans)

            # Shape: (batch_size, num_spans, emebedding_size + 2 * encoding_dim + feature_size)
            # span_embeddings = torch.cat([endpoint_span_embeddings, attended_span_embeddings], -1)
            span_embeddings = endpoint_span_embeddings
        else:
            span_embeddings = attended_span_embeddings

        # Prune based on mention scores.
        num_spans_to_keep = int(math.floor(self._spans_per_word * sequence_length))
        num_spans = spans.shape[1]
        num_spans_to_keep = min(num_spans_to_keep, num_spans)

        # Shape: (batch_size, num_spans)
        span_mention_scores = self._mention_scorer(
         self._mention_feedforward(span_embeddings)
        ).squeeze(-1)
        # Shape: (batch_size, num_spans) for all 3 tensors
        top_span_mention_scores, top_span_mask, top_span_indices = util.masked_topk(
            span_mention_scores, span_mask, num_spans_to_keep
        )
        verb_index = verb_indicator.argmax(1).unsqueeze(1).unsqueeze(2).repeat(1, 1, embedded_text_input.shape[-1])
        verb_embeddings = torch.gather(embedded_text_input, 1, verb_index)
        assert len(verb_embeddings.shape) == 3 and verb_embeddings.shape[1] == 1
        verb_embeddings = verb_embeddings.squeeze(1)
        # print(verb_indicator.sum(1, keepdim=True) > 0)
        verb_embeddings = torch.where((verb_indicator.sum(1, keepdim=True) > 0).repeat(1, verb_embeddings.shape[-1]), verb_embeddings, torch.zeros_like(verb_embeddings))
        # print(verb_embeddings)
        flat_top_span_indices = util.flatten_and_batch_shift_indices(top_span_indices, spans.shape[1])
        span_embeddings = util.batched_index_select(span_embeddings, top_span_indices, flat_top_span_indices)
        top_spans = util.batched_index_select(spans, top_span_indices, flat_top_span_indices)
        top_span_labels = util.batched_index_select(span_labels.unsqueeze(-1), top_span_indices, flat_top_span_indices).squeeze(-1)
        concatenated_span_embeddings = torch.cat((span_embeddings, verb_embeddings.unsqueeze(1).repeat(1, span_embeddings.shape[1], 1)), dim=2)
        # print(concatenated_span_embeddings[:,:,:])
        hidden = self.hidden_layer(concatenated_span_embeddings)
        # print(hidden[1,:,:])
        # print(top_span_indices)
        # print([[span_mention_scores[i,top_span_indices[i,j]].item() for j in range(top_span_indices.shape[1])] for i in range(top_span_labels.shape[0])])
        # print(top_span_mention_scores, self.vocab.get_token_index("O", namespace="span_labels"))
        predictions = self.output_layer(hidden)
        # predictions += top_span_mention_scores.unsqueeze(-1).repeat(1, 1, self.num_classes-1)
        predictions = torch.cat((torch.zeros_like(predictions[:,:,:1]), predictions), dim=-1)
        # print(top_span_mention_scores.unsqueeze(-1).repeat(1, 1, self.num_classes-1))

        output_dict = {}
        # We need to retain the mask in the output dictionary
        # so that we can crop the sequences to remove padding
        # when we do viterbi inference in self.make_output_human_readable.
        output_dict["mask"] = mask
        # We add in the offsets here so we can compute the un-wordpieced tags.
        words, verbs, offsets = zip(*[(x["words"], x["verb"], x["offsets"]) for x in metadata])
        output_dict["words"] = list(words)
        output_dict["verb"] = list(verbs)
        output_dict["wordpiece_offsets"] = list(offsets)

        if tags is not None:
            loss = (self._ce_loss(predictions.view(-1, predictions.shape[-1]), top_span_labels.view(-1))*top_span_mask.float().view(-1)).sum()/top_span_mask.float().sum()
            # print(top_span_labels)
            # print(predictions.argmax(-1))
            if not self.ignore_span_metric and self.span_metric is not None and not self.training:
                batch_verb_indices = [
                    example_metadata["verb_index"] for example_metadata in metadata
                ]
                batch_sentences = [example_metadata["words"] for example_metadata in metadata]
                # Get the BIO tags from make_output_human_readable()
                # TODO (nfliu): This is kind of a hack, consider splitting out part
                # of make_output_human_readable() to a separate function.
                batch_bio_predicted_tags = self.get_tags(top_spans, predictions, mask.shape[1], top_span_mask, output_dict)
                from allennlp_models.structured_prediction.models.srl import (
                    convert_bio_tags_to_conll_format,
                )

                batch_conll_predicted_tags = [
                    convert_bio_tags_to_conll_format(tags) for tags in batch_bio_predicted_tags
                ]
                batch_bio_gold_tags = [
                    example_metadata["gold_tags"] for example_metadata in metadata
                ]
                # print('G', batch_bio_gold_tags)
                batch_conll_gold_tags = [
                    convert_bio_tags_to_conll_format(tags) for tags in batch_bio_gold_tags
                ]
                self.span_metric(
                    batch_verb_indices,
                    batch_sentences,
                    batch_conll_predicted_tags,
                    batch_conll_gold_tags,
                )
            output_dict["loss"] = loss
        return output_dict

    def get_tags(self, spans, logits, sequence_length, span_mask, output_dict):
        predicted_tag_ids = logits.argmax(2)
        predicted_tags = []
        for i in range(spans.shape[0]):
            sequence = ["O" for _ in range(sequence_length)]
            for j in range(spans.shape[1]):
                if span_mask[i,j].item() == 0:
                    continue
                tag = predicted_tag_ids[i,j].item()
                if tag != self.vocab.get_token_index("O", namespace="span_labels"):
                    start = spans[i,j,0].item()
                    end = spans[i,j,1].item()
                    if all([el == "O" for el in sequence[start:end+1]]):
                        tag = self.vocab.get_token_from_index(tag, namespace="span_labels")
                        sequence[start] = "B-"+tag
                        for index in range(start+1, end+1):
                            sequence[index] = "I-"+tag
            predicted_tags.append([sequence[ind] for ind in output_dict["wordpiece_offsets"][i]])
        print(predicted_tags)
        return predicted_tags

    @overrides
    def make_output_human_readable(
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
        wordpiece_tags = []
        word_tags = []
        transition_matrix = self.get_viterbi_pairwise_potentials()
        start_transitions = self.get_start_transitions()
        # **************** Different ********************
        # We add in the offsets here so we can compute the un-wordpieced tags.
        for predictions, length, offsets in zip(
            predictions_list, sequence_lengths, output_dict["wordpiece_offsets"]
        ):
            max_likelihood_sequence, _ = viterbi_decode(
                predictions[:length], transition_matrix, allowed_start_transitions=start_transitions
            )
            tags = [
                self.vocab.get_token_from_index(x, namespace="labels")
                for x in max_likelihood_sequence
            ]

            wordpiece_tags.append(tags)
            word_tags.append([tags[i] for i in offsets])
        output_dict["wordpiece_tags"] = wordpiece_tags
        output_dict["tags"] = word_tags
        return output_dict

    def get_metrics(self, reset: bool = False):
        if self.ignore_span_metric:
            # Return an empty dictionary if ignoring the
            # span metric
            return {}

        else:
            metric_dict = self.span_metric.get_metric(reset=reset)

            # This can be a lot of metrics, as there are 3 per class.
            # we only really care about the overall metrics, so we filter for them here.
            return {x: y for x, y in metric_dict.items() if "overall" in x}

    def get_viterbi_pairwise_potentials(self):
        """
        Generate a matrix of pairwise transition potentials for the BIO labels.
        The only constraint implemented here is that I-XXX labels must be preceded
        by either an identical I-XXX tag or a B-XXX tag. In order to achieve this
        constraint, pairs of labels which do not satisfy this constraint have a
        pairwise potential of -inf.

        # Returns

        transition_matrix : `torch.Tensor`
            A `(num_labels, num_labels)` matrix of pairwise potentials.
        """
        all_labels = self.vocab.get_index_to_token_vocabulary("labels")
        num_labels = len(all_labels)
        transition_matrix = torch.zeros([num_labels, num_labels])

        for i, previous_label in all_labels.items():
            for j, label in all_labels.items():
                # I labels can only be preceded by themselves or
                # their corresponding B tag.
                if i != j and label[0] == "I" and not previous_label == "B" + label[1:]:
                    transition_matrix[i, j] = float("-inf")
        return transition_matrix

    def get_start_transitions(self):
        """
        In the BIO sequence, we cannot start the sequence with an I-XXX tag.
        This transition sequence is passed to viterbi_decode to specify this constraint.

        # Returns

        start_transitions : `torch.Tensor`
            The pairwise potentials between a START token and
            the first token of the sequence.
        """
        all_labels = self.vocab.get_index_to_token_vocabulary("labels")
        num_labels = len(all_labels)

        start_transitions = torch.zeros(num_labels)

        for i, label in all_labels.items():
            if label[0] == "I":
                start_transitions[i] = float("-inf")

        return start_transitions

    default_predictor = "semantic_role_labeling"
