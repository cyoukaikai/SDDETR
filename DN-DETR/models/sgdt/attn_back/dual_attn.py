# https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py

# ------------------------
import torch

try:
    from torch.overrides import has_torch_function, handle_torch_function
except:
    from torch._overrides import has_torch_function, handle_torch_function
Tensor = torch.Tensor

# temporarly comment out .is_nested
# from .attn import _mha_shape_check

# ============================
# (w_q, w_k, w_v) and (b_q, b_k, b_v) are packed, we cannot set requires_grad = False for only
# the weights (w_q, b_q) of q, or that of k, v. So we used some tricks to get around this issue.
# Still use them in the forward graph, but make their grad = 0, so they will never be updated.
# ============================


