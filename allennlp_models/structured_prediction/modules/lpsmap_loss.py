import torch

from allennlp.nn.util import sequence_cross_entropy_with_logits

class LpsmapLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, lpsmap_sol, gold, mask):
        ctx.save_for_backward(scores, lpsmap_sol, gold, mask)
        output = sequence_cross_entropy_with_logits(
            scores, gold, mask
        )
        return output
    @staticmethod
    def backward(ctx, grad_output):
        scores, lpsmap_sol, gold, mask = ctx.saved_tensors
        grad_lpsmapsol = grad_gold = grad_mask = None
        gold_expanded = torch.zeros_like(lpsmap_sol)
        gold_expanded.scatter_(2, gold.unsqueeze(-1), torch.ones_like(gold_expanded))
        grad_scores = (lpsmap_sol-gold_expanded)*mask.unsqueeze(-1).repeat(1, 1, gold_expanded.shape[-1]).float()*grad_output
        return grad_scores, grad_lpsmapsol, grad_gold, grad_mask
