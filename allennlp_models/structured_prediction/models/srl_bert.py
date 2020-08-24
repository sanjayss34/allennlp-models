from typing import Dict, List, Any, Union
import logging

from overrides import overrides
import torch
from torch.nn.modules import Linear, Dropout
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
# from transformers.modeling_bert import BertModel, BertConfig

from allennlp.common.checks import ConfigurationError
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, ConditionalRandomField
from allennlp.modules import ConditionalRandomField, FeedForward
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.modules.token_embedders import PretrainedTransformerMismatchedEmbedder, TokenEmbedder
from allennlp.nn import InitializerApplicator, util
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, viterbi_decode

from allennlp_models.structured_prediction.metrics.srl_eval_scorer import (
    DEFAULT_SRL_EVAL_PATH,
    SrlEvalScorer,
)

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from qpth.qp import QPFunction
from lpsmap import TorchFactorGraph, Xor, AtMostOne, Or, OrOut, Imply
from lpsmap.ad3qp.factor_graph import PFactorGraph
from lpsmap.ad3ext.sequence import PFactorSequence
from allennlp_models.structured_prediction.modules.lpsmap_loss import LpsmapLoss
from allennlp_models.structured_prediction.modules.new_lpsmap import LpSparseMap
from allennlp_models.structured_prediction.modules.torch_other_factor import TorchOtherFactor
# from new_lpsmap import lpsmap

logger = logging.getLogger(__name__)

@Model.register("srl_bert")
class SrlBert(Model):
    """

    A BERT based model [Simple BERT Models for Relation Extraction and Semantic Role Labeling (Shi et al, 2019)]
    (https://arxiv.org/abs/1904.05255) with some modifications (no additional parameters apart from a linear
    classification layer), which is currently the state-of-the-art single model for English PropBank SRL
    (Newswire sentences).

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
        bert_model: Union[str, AutoModel],
        mismatched_embedder: TokenEmbedder = None,
        lp: bool = False,
        lpsmap: bool = False,
        lpsmap_core_roles_only: bool = True,
        validation_inference: bool = True,
        batch_size: int = None,
        encoder: Seq2SeqEncoder = None,
        reinitialize_pos_embedding: bool = False,
        embedding_dropout: float = 0.0,
        mlp_hidden_size: int = 300,
        initializer: InitializerApplicator = InitializerApplicator(),
        label_smoothing: float = None,
        ignore_span_metric: bool = False,
        srl_eval_path: str = DEFAULT_SRL_EVAL_PATH,
        label_encoding: str = "BIO",
        constrain_crf_decoding: bool = None,
        include_start_end_transitions: bool = True,
        label_namespace: str = "labels",
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)

        if isinstance(bert_model, str):
            if mismatched_embedder is None:
                self.bert_model = AutoModel.from_pretrained(bert_model)
            self.bert_config = AutoConfig.from_pretrained(bert_model)
        else:
            if mismatched_embedder is None:
                self.bert_model = bert_model
            self.bert_config = bert_model.config
        if reinitialize_pos_embedding:
            self.bert_model._init_weights(self.bert_model.embeddings.position_embeddings)
            # self.bert_model._init_weights(self.bert_model.embeddings.token_type_embeddings)
        if mismatched_embedder is not None:
            self.bert_model = mismatched_embedder

        self._label_namespace = label_namespace
        self.num_classes = self.vocab.get_vocab_size(label_namespace)
        if srl_eval_path is not None:
            # For the span based evaluation, we don't want to consider labels
            # for verb, because the verb index is provided to the model.
            self.span_metric = SrlEvalScorer(srl_eval_path, ignore_classes=["V"])
        else:
            self.span_metric = None

        if constrain_crf_decoding is None:
            constrain_crf_decoding = label_encoding is not None

        self.label_encoding = label_encoding
        self.constrain_crf_decoding = constrain_crf_decoding
        if constrain_crf_decoding:
            if not label_encoding:
                raise ConfigurationError(
                    "constrain_crf_decoding is True, but no label_encoding was specified."
                )
            labels = self.vocab.get_index_to_token_vocabulary(label_namespace)
            constraints = allowed_transitions(label_encoding, labels)
        else:
            constraints = None

        self.include_start_end_transitions = include_start_end_transitions
        self.crf = ConditionalRandomField(
            self.num_classes, constraints, include_start_end_transitions=include_start_end_transitions
        )
        self._encoder = encoder
        representation_size = self.bert_config.hidden_size
        if self.bert_config.type_vocab_size == 1:
            representation_size = self.bert_config.hidden_size*2
        if encoder is None:
            self.tag_projection_layer = torch.nn.Sequential(
                                            Linear(representation_size, mlp_hidden_size),
                                            torch.nn.ReLU(),
                                            Linear(mlp_hidden_size, self.num_classes))
        else:
            self.tag_projection_layer = torch.nn.Sequential(
                                            Linear(encoder.get_output_dim()*2, mlp_hidden_size),
                                            torch.nn.ReLU(),
                                            Linear(mlp_hidden_size, self.num_classes))

        self.embedding_dropout = Dropout(p=embedding_dropout)
        self.predicate_embedding = torch.nn.Embedding(num_embeddings=2, embedding_dim=10)
        self._label_smoothing = label_smoothing
        self.ignore_span_metric = ignore_span_metric
        self._lp = lp
        self._lpsmap = lpsmap
        self._lpsmap_core_only = lpsmap_core_roles_only
        self._val_inference = validation_inference
        if self._lpsmap:
            self._core_roles = []
            for i in range(6):
                try:
                    self._core_roles.append(self.vocab.get_token_index("B-ARG"+str(i), namespace=self._label_namespace))
                except:
                    logger.info("B-ARG"+str(i)+" is not in labels")
            self._r_roles = []
            self._c_roles = []
            for i in range(self.num_classes):
                token = self.vocab.get_token_from_index(i, namespace=self._label_namespace)
                if token[:4] == "B-R-" and token[4:] != "ARG1":
                    try:
                        base_arg_index = self.vocab.get_token_index("B-"+token[4:], namespace=self._label_namespace)
                        self._r_roles.append((i, base_arg_index))
                    except:
                        logger.info("B-"+token[4:]+" is not in labels")
                elif token[:4] == "B-C-" and token[4:] != "ARG1":
                    try:
                        base_arg_index = self.vocab.get_token_index("B-"+token[4:], namespace=self._label_namespace)
                        self._c_roles.append((i, base_arg_index))
                    except:
                        logger.info("B-"+token[4:]+" is not in labels")
            # self._core_roles = [index for index in range(self.vocab.get_vocab_size("labels")) if index in [self.vocab.get_token_index("B-ARG"+str(i), namespace="labels") for i in range(3)]]
            self.lpsmap = None
        if lp:
            """self._layer_list = []
            self.length_map = {}
            self.lengths = []
            for max_sequence_length in [70, 100, 200, 300]:
                x = cp.Variable((max_sequence_length, self.vocab.get_vocab_size(namespace="labels")))
                S = cp.Parameter((max_sequence_length, self.vocab.get_vocab_size(namespace="labels")))
                constraints = [x >= 0, cp.sum(x, axis=1) == 1]
                objective = cp.Maximize(cp.sum(cp.multiply(x, S)))
                problem = cp.Problem(objective, constraints)
                assert problem.is_dpp()
                lp_layer = CvxpyLayer(problem, parameters=[S], variables=[x])
                self._layer_list.append(lp_layer)
                self.length_map[max_sequence_length] = len(self._layer_list)-1
                self.lengths.append(max_sequence_length)
            self._layer_list = torch.nn.ModuleList(self._layer_list)"""
            pass
        initializer(self)

    def forward(  # type: ignore
        self,
        tokens: TextFieldTensors,
        verb_indicator: torch.Tensor,
        sentence_end: torch.LongTensor,
        metadata: List[Any],
        tags: torch.LongTensor = None,
        offsets: torch.LongTensor = None
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
            metadata containing the original words in the sentence, the verb to compute the
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

        if isinstance(self.bert_model, PretrainedTransformerMismatchedEmbedder):
            encoder_inputs = tokens["tokens"]
            if self.bert_config.type_vocab_size > 1:
                encoder_inputs["type_ids"] = verb_indicator
            encoded_text = self.bert_model(**encoder_inputs)
            batch_size = encoded_text.shape[0]
            if self.bert_config.type_vocab_size == 1:
                verb_embeddings = encoded_text[torch.arange(batch_size).to(encoded_text.device),verb_indicator.argmax(1),:]
                verb_embeddings = torch.where((verb_indicator.sum(1, keepdim=True) > 0).repeat(1, verb_embeddings.shape[-1]), verb_embeddings, torch.zeros_like(verb_embeddings))
                encoded_text = torch.cat((encoded_text, verb_embeddings.unsqueeze(1).repeat(1, encoded_text.shape[1], 1)), dim=2)
            mask = tokens["tokens"]["mask"]
            index = mask.sum(1).argmax().item()
            # print(mask.shape, encoded_text.shape, tokens["tokens"]["token_ids"].shape, tags.shape, max([len(x['words']) for x in metadata]), mask.sum(1)[index].item())
            # print(tokens["tokens"]["token_ids"][index,:])
        else:
            mask = get_text_field_mask(tokens)
            bert_embeddings, _ = self.bert_model(
                input_ids=util.get_token_ids_from_text_field_tensors(tokens),
                # token_type_ids=verb_indicator,
                attention_mask=mask,
            )

            batch_size, _ = mask.size()
            embedded_text_input = self.embedding_dropout(bert_embeddings)
            # Restrict to sentence part
            sentence_mask = (torch.arange(mask.shape[1]).unsqueeze(0).repeat(batch_size, 1).to(mask.device) < sentence_end.unsqueeze(1).repeat(1, mask.shape[1])).long()
            cutoff = sentence_end.max().item()
            if self._encoder is None:
                encoded_text = embedded_text_input
                mask = sentence_mask[:,:cutoff].contiguous()
                encoded_text = encoded_text[:,:cutoff,:]
                tags = tags[:,:cutoff].contiguous()
            else:
                predicate_embeddings = self.predicate_embedding(verb_indicator)
                encoder_inputs = torch.cat((embedded_text_input, predicate_embeddings), dim=-1)
                encoded_text = self._encoder(encoder_inputs, mask=sentence_mask.bool())
                # print(verb_indicator)
                predicate_index = (verb_indicator*torch.arange(start=verb_indicator.shape[-1]-1, end=-1, step=-1).to(mask.device).unsqueeze(0).repeat(batch_size, 1)).argmax(1)
                # print(predicate_index)
                predicate_hidden = encoded_text[torch.arange(batch_size).to(mask.device),predicate_index]
                predicate_exists, _ = verb_indicator.max(1)
                encoded_text = encoded_text[:,:cutoff,:]
                tags = tags[:,:cutoff].contiguous()
                mask = sentence_mask[:,:cutoff].contiguous()
                predicate_exists = predicate_exists.unsqueeze(1).repeat(1, encoded_text.shape[-1])
                predicate_hidden = torch.where(predicate_exists > 0, predicate_hidden, torch.zeros_like(predicate_hidden))
                encoded_text = torch.cat((encoded_text, predicate_hidden.unsqueeze(1).repeat(1, encoded_text.shape[1], 1)), dim=-1)

        sequence_length = encoded_text.shape[1]
        logits = self.tag_projection_layer(encoded_text)
        # print(mask, logits)
        if self._lp and sequence_length <= 100:
            eps = 1e-4
            Q = eps*torch.eye(sequence_length*self.num_classes, sequence_length*self.num_classes).unsqueeze(0).repeat(batch_size, 1, 1).to(logits.device).float()
            p = logits.view(batch_size, -1)
            G = -1*torch.eye(sequence_length*self.num_classes).unsqueeze(0).repeat(batch_size, 1, 1).to(logits.device).float()
            h = torch.zeros_like(p)
            A = torch.arange(sequence_length*self.num_classes).unsqueeze(0).repeat(sequence_length, 1)
            A2 = torch.arange(sequence_length).unsqueeze(1).repeat(1, sequence_length*self.num_classes)*self.num_classes
            A = torch.where((A >= A2) & (A < A2+self.num_classes), torch.ones_like(A), torch.zeros_like(A))
            A = A.unsqueeze(0).repeat(batch_size, 1, 1).to(logits.device).float()
            b = torch.ones_like(A[:,:,0])
            probs = QPFunction()(Q, p, torch.autograd.Variable(torch.Tensor()), torch.autograd.Variable(torch.Tensor()), A, b)
            probs = probs.view(batch_size, sequence_length, self.num_classes)
            """logits_shape = logits.shape
            logits = torch.where(mask.bool().unsqueeze(-1).repeat(1, 1, logits.shape[-1]), logits, logits-10000)
            max_sequence_length = min([l for l in self.lengths if l >= sequence_length])
            if max_sequence_length > logits_shape[1]:
                logits = torch.cat((logits, torch.zeros((batch_size, max_sequence_length-logits_shape[1], logits_shape[2])).to(logits.device)), dim=1)
            lp_layer = self._layer_list[self.length_map[max_sequence_length]]
            probs, = lp_layer(logits)
            print(torch.isnan(probs).any())
            if max_sequence_length > logits_shape[1]:
                probs = probs[:,:logits_shape[1],:]"""
            logits = (torch.nn.functional.relu(probs)+1e-4).log()
        if self._lpsmap:
            if self._lpsmap_core_only:
                all_logits = logits
            else:
                all_logits = torch.cat((logits, 0.5*torch.ones((batch_size, 1, logits.shape[-1])).to(logits.device)), dim=1)
            probs = []
            for i in range(batch_size):
                if self.constrain_crf_decoding:
                    unaries = logits[i,:,:].view(-1).cpu()
                    additionals = self.crf.transitions.view(-1).repeat(sequence_length)+10000*(self.crf._constraint_mask[:-2,:-2]-1).view(-1).repeat(sequence_length)
                    start_transitions = self.crf.start_transitions+10000*(self.crf._constraint_mask[-2,:-2]-1)
                    end_transitions = self.crf.start_transitions+10000*(self.crf._constraint_mask[-1,:-2]-1)
                    additionals = torch.cat((additionals, start_transitions, end_transitions), dim=0).cpu()
                    fg = TorchFactorGraph()
                    x = fg.variable_from(unaries)
                    f = PFactorSequence()

                    f.initialize([self.num_classes for _ in range(sequence_length)])
                    factor = TorchOtherFactor(f, x, additionals)
                    fg.add(factor)
                    # add budget constraint for each state
                    for state in self._core_roles:
                        vars_state = x[state::self.num_classes]
                        fg.add(AtMostOne(vars_state))
                    # solve SparseMAP
                    fg.solve(max_iter=200)
                    probs.append(unaries.to(logits.device).view(sequence_length, self.num_classes))
                else:
                    fg = TorchFactorGraph()
                    x = fg.variable_from(all_logits[i,:,:].cpu())
                    for j in range(sequence_length):
                        fg.add(Xor(x[j,:]))
                    for j in self._core_roles:
                        fg.add(AtMostOne(x[:sequence_length,j]))
                    if not self._lpsmap_core_only:
                        full_sequence = list(range(sequence_length))
                        base_roles = set([second for (_, second) in self._r_roles+self._c_roles])
                        """for (r_role, base_role) in self._r_roles+self._c_roles:
                            for j in range(sequence_length):
                                fg.add(Imply(x[full_sequence+[j],[base_role]*sequence_length+[r_role]], negated=[True]*(sequence_length+1)))"""
                        for base_role in base_roles:
                            fg.add(OrOut(x[:,base_role]))
                        for (r_role, base_role) in self._r_roles+self._c_roles:
                            fg.add(OrOut(x[:,r_role]))
                            fg.add(Or(x[[sequence_length, sequence_length], [r_role, base_role]], negated=[True, False]))
                    max_iter = 100
                    if not self._lpsmap_core_only:
                        max_iter = min(max_iter, 400)
                    elif (not self.training) and not self._val_inference:
                        max_iter = min(max_iter, 200)
                    fg.solve(max_iter=max_iter)
                    probs.append(x.value[:sequence_length,:].contiguous().to(logits.device))
            class_probabilities = torch.stack(probs)
            # class_probabilities = self.lpsmap(logits)
            max_seq_length = 200
            # if self.lpsmap is None:
            """with torch.no_grad():
                # self.lpsmap = LpSparseMap(num_rows=sequence_length, num_cols=self.num_classes, batch_size=batch_size, device=logits.device, constraints=[('xor', ('row', list(range(sequence_length)))), ('budget', ('col', self._core_roles))])
                max_iter = 1000
                constraint_types = ["xor", "budget"]
                constraint_dims = ["row", "col"]
                constraint_sets = [list(range(sequence_length)), self._core_roles]
                class_probabilities = lpsmap(logits, constraint_types, constraint_dims, constraint_sets, max_iter)
                # if max_seq_length > sequence_length:
                #     logits = torch.cat((logits, -9999.*torch.ones((batch_size, max_seq_length-sequence_length, self.num_classes)).to(logits.device)), dim=1)
                # class_probabilities = self.lpsmap.solve(logits, max_iter=max_iter)"""
            # logits = (class_probabilities+1e-4).log()
        else:
            reshaped_log_probs = logits.view(-1, self.num_classes)
            class_probabilities = F.softmax(reshaped_log_probs, dim=-1).view(
                [batch_size, sequence_length, self.num_classes]
            )
        output_dict = {"logits": logits, "class_probabilities": class_probabilities}
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
            # print(mask.shape, tags.shape, logits.shape, tags.max(), tags.min())
            if self._lpsmap:
                loss = LpsmapLoss.apply(logits, class_probabilities, tags, mask)
                # tags_1hot = torch.zeros_like(class_probabilities).scatter_(2, tags.unsqueeze(-1), torch.ones_like(class_probabilities))
                # loss = -(tags_1hot*class_probabilities*mask.unsqueeze(-1).repeat(1, 1, class_probabilities.shape[-1])).sum()
            elif self.constrain_crf_decoding:
                loss = -self.crf(logits, tags, mask)
            else:
                loss = sequence_cross_entropy_with_logits(
                    logits, tags, mask, label_smoothing=self._label_smoothing
                )
            if not self.ignore_span_metric and self.span_metric is not None and not self.training:
                batch_verb_indices = [
                    example_metadata["verb_index"] for example_metadata in metadata
                ]
                batch_sentences = [example_metadata["words"] for example_metadata in metadata]
                # Get the BIO tags from make_output_human_readable()
                # TODO (nfliu): This is kind of a hack, consider splitting out part
                # of make_output_human_readable() to a separate function.
                batch_bio_predicted_tags = self.make_output_human_readable(output_dict).pop("tags")
                from allennlp_models.structured_prediction.models.srl import (
                    convert_bio_tags_to_conll_format,
                )

                if self.constrain_crf_decoding and not self._lpsmap:
                    batch_conll_predicted_tags = [
                        convert_bio_tags_to_conll_format([self.vocab.get_token_from_index(tag, namespace=self._label_namespace) for tag in seq]) for (seq, _) in self.crf.viterbi_tags(logits, mask)
                    ]
                else:
                    batch_conll_predicted_tags = [
                        convert_bio_tags_to_conll_format(tags) for tags in batch_bio_predicted_tags
                    ]
                batch_bio_gold_tags = [
                    example_metadata["gold_tags"] for example_metadata in metadata
                ]
                # print(batch_bio_gold_tags)
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
            output_dict["gold_tags"] = [x["gold_tags"] for x in metadata]
        return output_dict

    def lpsmap(self, scores: torch.Tensor):
        batch_size, sequence_length, num_classes = scores.shape
        C_indices = torch.arange(sequence_length*num_classes).to(scores.device)
        batch_range = torch.arange(batch_size).to(scores.device)
        C_indices = torch.cat((batch_range.unsqueeze(1).repeat(1, C_indices.numel()).view(-1).unsqueeze(0),
                               C_indices.repeat(batch_size).unsqueeze(0).repeat(2, 1)),
                              dim=0)
        C_values = torch.ones_like(C_indices[0,:]).float()
        C = torch.sparse.FloatTensor(C_indices, C_values, torch.Size([batch_size, sequence_length*num_classes, sequence_length*num_classes])).to(scores.device)
        delta = torch.ones_like(scores).view(batch_size, -1, 1)
        M = C.clone()
        D = torch.bmm(C, delta) # [batch_size, num_factors*num_variables_per_factor, 1]
        D_inverse = 1./D
        D_inverse_per_factor = D_inverse.view(-1, num_classes, 1) # [batch_size*num_factors, num_variables_per_factor, 1]
        # D_block = torch.diag_embed(D)
        # D_inverse_block = torch.diag_embed(D_inverse)
        # M_tilde = torch.bmm(M, D_inverse).t()
        mu = torch.ones_like(scores).view(batch_size, -1, 1)
        lambd = torch.zeros((batch_size*sequence_length, num_classes, 1)).to(scores.device)
        gamma = 1
        T = 10
        # Reshape D, eta, and C_tilde so that a batch has only the variables for a single factor
        eta = scores.view(batch_size*sequence_length, num_classes, 1)
        D = D.view(batch_size*sequence_length, num_classes, 1)
        D_eta_product = D*eta # [batch_size*num_factors, num_variables_per_factor]
        # C_tilde = C_tilde.view(batch_size*sequence_length, num_classes, sequence_length*num_classes)
        range_per_factor = torch.arange(num_classes).to(scores.device).unsqueeze(0).repeat(batch_size*sequence_length, 1).float()
        eps = 1e-6
        for t in range(T):
            Cmu = torch.bmm(C, mu).view(-1, num_classes, 1) # [batch_size*num_factors, num_variables_per_factor, 1]
            eta_tilde = (D_eta_product-lambd+gamma*D_inverse_per_factor*Cmu)/(gamma+1) # [batch_size*num_factors, num_variables_per_factor, 1]
            # Procedure for solving XOR QP according to Duchi et al. 2008
            # TODO: adjust this for when Mf_tilde \neq I (i.e. when there are more factors)
            eta_tilde_sorted, eta_indices = eta_tilde.view(batch_size*sequence_length, -1).sort(dim=-1, descending=True) # [batch_size*num_factors, num_variables_per_factor], [batch_size*num_factors, num_variables_per_factor]
            eta_tilde_cumsum = torch.cumsum(eta_tilde_sorted, dim=1) # [batch_size*num_factors, num_variables_per_factor]
            eta_tilde_cumsum = (eta_tilde_cumsum-1)/(range_per_factor+1) # [batch_size*num_factors, num_variables_per_factor]
            rho = ((eta_tilde_sorted-eta_tilde_cumsum > 0).float()*(1+range_per_factor)).argmax(1) # [batch_size*num_factors]
            theta = torch.gather(eta_tilde_cumsum, 1, rho.unsqueeze(-1)).repeat(1, num_classes).unsqueeze(-1) #[batch_size*num_factors, num_variables_per_factor, 1]
            p = torch.nn.functional.relu(eta_tilde-theta) # [batch_size*num_factors, num_variables_per_factor = num_assignments_per_factor, 1]
            Mp = torch.bmm(M, p.view(batch_size, -1, 1)) # [batch_size, num_factors*num_variables_per_factor, 1]
            Dinv_Mp = D_inverse*Mp # [batch_size, num_factors*num_variables_per_factor, 1]
            Dinv2_Mp = D_inverse*Dinv_Mp # [batch_size, num_factors*num_variables_per_factor, 1]
            mu_new = torch.bmm(C.transpose(1, 2), Dinv2_Mp) # [batch_size, num_variables, 1]
            lambd_diff = D_inverse*torch.bmm(C, mu_new)-Dinv_Mp # [batch_size, num_factors*num_variables_per_factor, 1]
            lambd += gamma*lambd_diff.view(-1, num_classes, 1) # [batch_size*num_factors, num_variables_per_factor]
            mu_diff_norm = torch.norm((mu_new-mu).squeeze(-1), dim=1) # [batch_size]
            lambd_diff_norm = torch.norm(lambd_diff.squeeze(-1), dim=1) # [batch_size]
            if mu_diff_norm.max().item() < eps and lambd_diff_norm.max().item() < eps:
                break
            mu = mu_new
        mu = mu.view(batch_size, sequence_length, num_classes)
        return mu

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
                self.vocab.get_token_from_index(x, namespace=self._label_namespace)
                for x in max_likelihood_sequence
            ]

            wordpiece_tags.append(tags)
            if isinstance(self.bert_model, PretrainedTransformerMismatchedEmbedder):
                word_tags.append(tags)
            else:
                word_tags.append([tags[i] for i in offsets])
            # print(word_tags)
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
        all_labels = self.vocab.get_index_to_token_vocabulary(self._label_namespace)
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
        all_labels = self.vocab.get_index_to_token_vocabulary(self._label_namespace)
        num_labels = len(all_labels)

        start_transitions = torch.zeros(num_labels)

        for i, label in all_labels.items():
            if label[0] == "I":
                start_transitions[i] = float("-inf")

        return start_transitions

    default_predictor = "semantic_role_labeling"
