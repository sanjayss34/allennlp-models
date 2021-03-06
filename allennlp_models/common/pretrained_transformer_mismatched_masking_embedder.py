from typing import Optional, Dict, Any, List

from overrides import overrides
import torch

from allennlp.modules.token_embedders import PretrainedTransformerEmbedder, TokenEmbedder
from allennlp.nn import util

from allennlp_models.common.pretrained_transformer_masking_embedder import PretrainedTransformerMaskingEmbedder

@TokenEmbedder.register("pretrained_transformer_mismatched_masking")
class PretrainedTransformerMaskingMismatchedEmbedder(TokenEmbedder):
    """
    Use this embedder to embed wordpieces given by `PretrainedTransformerMismatchedIndexer`
    and to pool the resulting vectors to get word-level representations.

    Registered as a `TokenEmbedder` with name "pretrained_transformer_mismatched".

    # Parameters

    model_name : `str`
        The name of the `transformers` model to use. Should be the same as the corresponding
        `PretrainedTransformerMismatchedIndexer`.
    max_length : `int`, optional (default = `None`)
        If positive, folds input token IDs into multiple segments of this length, pass them
        through the transformer model independently, and concatenate the final representations.
        Should be set to the same value as the `max_length` option on the
        `PretrainedTransformerMismatchedIndexer`.
    train_parameters: `bool`, optional (default = `True`)
        If this is `True`, the transformer weights get updated during training.
    last_layer_only: `bool`, optional (default = `True`)
        When `True` (the default), only the final layer of the pretrained transformer is taken
        for the embeddings. But if set to `False`, a scalar mix of all of the layers
        is used.
    gradient_checkpointing: `bool`, optional (default = `None`)
        Enable or disable gradient checkpointing.
    tokenizer_kwargs: `Dict[str, Any]`, optional (default = `None`)
        Dictionary with
        [additional arguments](https://github.com/huggingface/transformers/blob/155c782a2ccd103cf63ad48a2becd7c76a7d2115/transformers/tokenization_utils.py#L691)
        for `AutoTokenizer.from_pretrained`.
    transformer_kwargs: `Dict[str, Any]`, optional (default = `None`)
        Dictionary with
        [additional arguments](https://github.com/huggingface/transformers/blob/155c782a2ccd103cf63ad48a2becd7c76a7d2115/transformers/modeling_utils.py#L253)
        for `AutoModel.from_pretrained`.
    """  # noqa: E501

    def __init__(
        self,
        model_name: str,
        max_length: int = None,
        train_parameters: bool = True,
        last_layer_only: bool = True,
        gradient_checkpointing: Optional[bool] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        transformer_kwargs: Optional[Dict[str, Any]] = None,
        mask_probability: float = 0.15,
    ) -> None:
        super().__init__()
        # The matched version v.s. mismatched
        self._matched_embedder = PretrainedTransformerMaskingEmbedder(
            model_name,
            max_length=max_length,
            train_parameters=train_parameters,
            last_layer_only=last_layer_only,
            gradient_checkpointing=gradient_checkpointing,
            tokenizer_kwargs=tokenizer_kwargs,
            transformer_kwargs=transformer_kwargs,
        )
        self._mask_probability = 0.15
        self._mask_token_probability = 0.8
        self._mask_random_probability = 0.1

    @overrides
    def get_output_dim(self):
        return self._matched_embedder.get_output_dim()

    @overrides
    def forward(
        self,
        token_ids: torch.LongTensor,
        mask: torch.BoolTensor,
        offsets: torch.LongTensor,
        wordpiece_mask: torch.BoolTensor,
        type_ids: Optional[torch.LongTensor] = None,
        segment_concat_mask: Optional[torch.BoolTensor] = None,
        masked_lm: Optional[List[bool]] = None
    ) -> torch.Tensor:  # type: ignore
        """
        # Parameters

        token_ids: `torch.LongTensor`
            Shape: [batch_size, num_wordpieces] (for exception see `PretrainedTransformerEmbedder`).
        mask: `torch.BoolTensor`
            Shape: [batch_size, num_orig_tokens].
        offsets: `torch.LongTensor`
            Shape: [batch_size, num_orig_tokens, 2].
            Maps indices for the original tokens, i.e. those given as input to the indexer,
            to a span in token_ids. `token_ids[i][offsets[i][j][0]:offsets[i][j][1] + 1]`
            corresponds to the original j-th token from the i-th batch.
        wordpiece_mask: `torch.BoolTensor`
            Shape: [batch_size, num_wordpieces].
        type_ids: `Optional[torch.LongTensor]`
            Shape: [batch_size, num_wordpieces].
        segment_concat_mask: `Optional[torch.BoolTensor]`
            See `PretrainedTransformerEmbedder`.

        # Returns

        `torch.Tensor`
            Shape: [batch_size, num_orig_tokens, embedding_size].
        """
        masked_lm_labels = -100*torch.ones_like(token_ids)
        masked_token_ids = token_ids
        activate_masking = masked_lm is not None and any(masked_lm)
        if activate_masking:
            batch_size, num_orig_tokens = mask.shape
            masked_lm = torch.tensor(masked_lm, dtype=torch.bool).to(token_ids.device)
            mask_probs = torch.rand(mask.shape, device=mask.device)
            mask_token_choices = (mask_probs < self._mask_probability*self._mask_token_probability) & mask & masked_lm.unsqueeze(-1)
            mask_random_choices = (mask_probs >= self._mask_probability*self._mask_token_probability) & (mask_probs < self._mask_probability*(self._mask_token_probability+self._mask_random_probability)) & mask & masked_lm.unsqueeze(-1)
            all_mask_choices = (mask_probs < self._mask_probability) & mask & masked_lm.unsqueeze(-1)
            mask_token_indices = mask_token_choices.nonzero()
            mask_random_indices = mask_random_choices.nonzero()
            mask_random_values = torch.randint(low=0, high=self._matched_embedder.transformer_model.config.vocab_size, size=token_ids.shape, device=mask.device)
            all_mask_indices = all_mask_choices.nonzero()
            masked_token_ids = token_ids.clone()
            for i in range(mask_token_indices.shape[0]):
                offset_start_end = offsets[mask_token_indices[i][0].item(), mask_token_indices[i][1].item(),:]
                masked_token_ids[mask_token_indices[i][0].item(), offset_start_end[0].item():offset_start_end[1].item()+1] = self._matched_embedder._mask_token_id
            for i in range(mask_random_indices.shape[0]):
                offset_start_end = offsets[mask_random_indices[i][0].item(), mask_random_indices[i][1].item(),:]
                masked_token_ids[mask_random_indices[i][0].item(), offset_start_end[0].item():offset_start_end[1].item()+1] = mask_random_values[mask_random_indices[i][0].item(), offset_start_end[0].item():offset_start_end[1].item()+1]
            for i in range(all_mask_indices.shape[0]):
                offset_start_end = offsets[all_mask_indices[i][0].item(), all_mask_indices[i][1].item(),:]
                masked_lm_labels[all_mask_indices[i][0], offset_start_end[0].item():offset_start_end[1].item()+1] = token_ids[all_mask_indices[i][0], offset_start_end[0].item():offset_start_end[1].item()+1]
        # Shape: [batch_size, num_wordpieces, embedding_size].
        embeddings, masked_lm_loss = self._matched_embedder(
            masked_token_ids, wordpiece_mask, type_ids=type_ids, segment_concat_mask=segment_concat_mask, masked_lm_labels=masked_lm_labels
        )

        # span_embeddings: (batch_size, num_orig_tokens, max_span_length, embedding_size)
        # span_mask: (batch_size, num_orig_tokens, max_span_length)
        span_embeddings, span_mask = util.batched_span_select(embeddings.contiguous(), offsets)
        span_mask = span_mask.unsqueeze(-1)
        span_embeddings *= span_mask  # zero out paddings

        span_embeddings_sum = span_embeddings.sum(2)
        span_embeddings_len = span_mask.sum(2)
        # Shape: (batch_size, num_orig_tokens, embedding_size)
        orig_embeddings = span_embeddings_sum / torch.clamp_min(span_embeddings_len, 1)

        # All the places where the span length is zero, write in zeros.
        orig_embeddings[(span_embeddings_len == 0).expand(orig_embeddings.shape)] = 0

        if activate_masking:
            return orig_embeddings, masked_lm_loss
        return orig_embeddings
