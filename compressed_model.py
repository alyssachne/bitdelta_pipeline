import torch
from torch import nn, optim

import huggingface_hub


class CompressedModel(nn.Module):
    def __init__(self, base_model, finetuned_model):
        """
        Precondition: base_model and finetuned_model are of the same architecture
        """
        super(CompressedModel, self).__init__()
        # for each layer in both model
        # compute the difference between them
        for layer in base_model.state_dict().keys():
            base_weights = base_model.state_dict()[layer]
            finetuned_weights = finetuned_model.state_dict()[layer]
            # convert weights to positive/negative mask
            # 1 if finetuned weight is greater than base weight
            # -1 otherwise
            diff = torch.where(finetuned_weights >= base_weights, 1.0, -1.0)
            # store the mask in the compressed model
            self.register_parameter(layer + "_mask", nn.Parameter(diff))
            # also, add the trainable bitdelta to parameters
            self.register_parameter(layer + "_bitdelta", nn.Parameter(torch.tensor(0.5)))

    def forward(self, x, base_model):
        # for each layer in the model
        for layer in self.state_dict().keys():
            base_weights = base_model.state_dict()[layer]
            mask = self.state_dict()[layer + "_mask"]
            bitdelta = self.state_dict()[layer + "_bitdelta"]
            # compute the compressed weights
            compressed_weights = base_weights + bitdelta * mask
            # apply the compressed weights to the input
            x = torch.matmul(x, compressed_weights)
        return x
