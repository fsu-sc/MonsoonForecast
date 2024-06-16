import torch.nn.functional as F
import torch

def nll_loss(output, target):
    # Ensure the output tensor is of type Float and contains log-probabilities
    if output.dtype != torch.float32:
        output = output.float()
    output = F.log_softmax(output, dim=1)
    
    # Ensure the target tensor is of type Long
    if target.dtype != torch.long:
        target = target.long()
    
    return F.nll_loss(output, target)
