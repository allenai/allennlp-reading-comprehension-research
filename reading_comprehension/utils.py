import torch


def memory_effient_masked_softmax(vector: torch.Tensor, mask: torch.Tensor,
                                  dim: int = -1, mask_value=-1e7) -> torch.Tensor:
    """
    This is an approximate version of `allennlp.nn.util.masked_softmax`.
    By using less operations here than the original `masked_softmax`, we save a lot of memory.
    But you should be careful that this function does not return an array of ``0.0``, as the
    original `mask_softmax` does, in the case that the input vector is completely masked.
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # To limit numerical errors from large vector elements outside the mask, we zero these out.
        result = torch.nn.functional.softmax(vector + (1 - mask) * mask_value, dim=dim)
    return result
