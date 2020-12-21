import torch

# from allennlp.nn.util import sequence_cross_entropy_with_logits

class LpsmapLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, lpsmap_sol, gold, mask):
        loss_func = torch.nn.BCEWithLogitsLoss(reduction='none')
        ctx.save_for_backward(scores, lpsmap_sol, gold, mask)
        output = (loss_func(scores, gold.float())*mask.float()).sum()/mask.sum().float()
        return output
    @staticmethod
    def backward(ctx, grad_output):
        scores, lpsmap_sol, gold, mask = ctx.saved_tensors
        grad_lpsmapsol = grad_gold = grad_mask = None
        grad_scores = (lpsmap_sol-gold.float()).float()*grad_output
        if mask is not None:
            grad_scores *= mask.float()
        return grad_scores, grad_lpsmapsol, grad_gold, grad_mask
