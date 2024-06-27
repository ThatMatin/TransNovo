import torch

def mean_batch_acc(logits: torch.Tensor, labels: torch.Tensor):
    """
    Note that labels must be already aligned
    """
    batch_size = labels.size(0)
    batch_accs = torch.zeros(batch_size)
    idxs = logits.argmax(-1)
    correct = idxs == labels
    # trim after EOS
    EOS_idx = (labels == 2).nonzero(as_tuple=True)[1]
    for i in range(batch_size):
        batch_accs[i] = correct[i, :EOS_idx[i]].float().mean()
    return batch_accs.mean()
