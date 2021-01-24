import numpy as np
import torch

from new_lpsmap import lpsmap_forward, lpsmap_backward, lpsmap_setup

class LpsmapFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, variable_selector_list, constraint_sets, constraint_types, constraint_sizes, budgets, negated, variable_degrees, max_iter):
        outputs = lpsmap_forward(scores, variable_selector_list, constraint_sets, constraint_types, constraint_sizes, budgets, negated, variable_degrees, max_iter)
        ctx.args_for_backward = [outputs[1], variable_selector_list, constraint_sets, constraint_types, constraint_sizes, budgets, negated, variable_degrees, max_iter]
        return outputs[0]
    @staticmethod
    def backward(ctx, grad_outputs):
        grad_scores = lpsmap_backward(grad_outputs, *ctx.args_for_backward)
        print(grad_scores)
        return grad_scores, None, None, None, None, None, None, None, None

class BatchLpsmap(torch.nn.Module):
    def __init__(self,
                 num_variables,
                 constraint_types,
                 constraint_sizes,
                 constraint_sets,
                 budgets, # Only used for budget constraints
                 negated,
                 constraint_coefficients,
                 device,
                 batch_size,
                 max_iter=1000):
        super().__init__()
        self.constraint_types = constraint_types
        self.constraint_sizes = constraint_sizes
        self.constraint_sets = constraint_sets
        self.budgets = budgets
        self.negated = [[int(val) for val in lst] for lst in negated]
        self.constraint_coefficients = constraint_coefficients
        self.num_variables = num_variables
        self.max_iter = max_iter
        self.device = device
        self.variable_degrees = [[0 for _ in range(num_variables)]]
        for constraint in constraint_sets:
            for var in constraint:
                self.variable_degrees[0][var] += 1
        self.variable_degrees = torch.tensor(self.variable_degrees, dtype=torch.float32, device=device)
        self.variable_degrees = self.variable_degrees.repeat(batch_size, 1)
        self.variable_selector_list = []
        self.batch_size = batch_size
        self.variable_selector_list = self.construct_variable_selectors(constraint_sets, batch_size)
        self.lpsmap_inputs = self.setup_lpsmap_inputs(self.variable_selector_list, batch_size)
        self.smaller_variable_selectors = {}
        self.smaller_lpsmap_inputs = {}

    def construct_variable_selectors(self, constraint_sets, batch_size):
        batch_range = torch.arange(batch_size).to(self.device).long()
        variable_selector_list = []
        for i in range(len(constraint_sets)):
            row_indices_list = [j for j in range(len(constraint_sets[i]))]
            row_indices = torch.tensor(row_indices_list, dtype=torch.int64, device=self.device)
            col_indices = torch.LongTensor(constraint_sets[i]).to(self.device).long()
            var_selector_indices = torch.cat((batch_range.unsqueeze(1).repeat(1, col_indices.numel()).view(1, -1),
                                              row_indices.unsqueeze(0).repeat(batch_size, 1).view(1, -1),
                                              col_indices.unsqueeze(0).repeat(batch_size, 1).view(1, -1)), dim=0)
            var_selector_values = torch.ones_like(var_selector_indices[0,:]).float()
            var_selector = torch.sparse.FloatTensor(var_selector_indices, var_selector_values, torch.Size([batch_size, len(constraint_sets[i]), self.num_variables])).to(self.device)
            variable_selector_list.append(var_selector)
        return variable_selector_list

    def setup_lpsmap_inputs(self, variable_selector_list, batch_size):
        variable_degrees = self.variable_degrees[:batch_size,:].contiguous()
        lpsmap_inputs = lpsmap_setup(batch_size, self.num_variables, variable_selector_list, self.constraint_sets, self.constraint_types, self.constraint_sizes, self.negated, self.constraint_coefficients, variable_degrees)
        lpsmap_inputs = [variable_selector_list]+list(lpsmap_inputs)+[self.constraint_sets, self.constraint_types, self.constraint_sizes, self.budgets, self.negated, variable_degrees, self.max_iter]
        return list(lpsmap_inputs)

    def forward(self, scores: torch.Tensor):
        batch_size = scores.shape[0]
        assert batch_size <= self.batch_size
        if batch_size < self.batch_size:
            if batch_size in self.smaller_variable_selectors:
                variable_selector_list = self.smaller_variable_selectors[batch_size]
                lpsmap_inputs = self.smaller_lpsmap_inputs[batch_size]
            else:
                variable_selector_list = self.construct_variable_selectors(self.constraint_sets, batch_size)
                self.smaller_variable_selectors[batch_size] = variable_selector_list
                lpsmap_inputs = self.setup_lpsmap_inputs(variable_selector_list, batch_size)
                self.smaller_lpsmap_inputs[batch_size] = lpsmap_inputs
            variable_degrees = self.variable_degrees[:batch_size,:].contiguous()
        else:
            variable_selector_list = self.variable_selector_list
            variable_degrees = self.variable_degrees
            lpsmap_inputs = self.lpsmap_inputs
        # result = lpsmap(scores, variable_selector_list, self.constraint_sets, self.constraint_types, self.constraint_sizes, self.budgets, self.negated, variable_degrees, self.max_iter)
        # result = LpsmapFunction.apply(scores, variable_selector_list, self.constraint_sets, self.constraint_types, self.constraint_sizes, self.budgets, self.negated, variable_degrees, self.max_iter)
        result = lpsmap_forward(*([scores.float()]+lpsmap_inputs))[0]
        return result
