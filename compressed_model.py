import torch
from torch import nn, optim
import gc

import huggingface_hub

from binary_gemm_kernel import pack, unpack


class BinaryDiff(nn.Module):
    def __init__(self, base, finetune):
        super().__init__()
        diff = finetune - base
        quantile = diff.float().abs().mean()

        # mask = torch.ones_like(diff)
        # mask[diff < 0] = 0
        # mask = pack(mask.bool().T)

        # self.register_buffer("mask", mask)
        self.register_buffer("base", base.T)
        self.register_parameter(
            "coeff", # this is our bitdelta
            nn.Parameter(
                torch.tensor(
                    quantile,
                    dtype=torch.float32,
                    requires_grad=True,
                    device=base.device,
                )
            ),
        )
        if diff.is_meta:
            print("Warning: diff is meta")
        if base.is_meta:
            print("Warning: base is meta")
        if finetune.is_meta:
            print("Warning: finetune is meta")
        del base, finetune, diff

    def forward(self, x):
        # repeated_mask = self.mask.unsqueeze(0).repeat(x.size(0), 1, 1)
        return x @ self.base + self.coeff * x @ self.base

def compress_diff(base_model, finetuned_model, finetuned_compressed_model, device):
    def compress_module(name, subname, module, submodule, device):
        # target_device = submodule.weight.device
        
        base_weight = base_model.get_submodule(f"{name}.{subname}").weight.detach()
        if base_weight.is_meta:
            base_weight = torch.zeros(base_weight.size())
        base_weight = base_weight.to(device)
        finetuned_weight = finetuned_model.get_submodule(f"{name}.{subname}").weight.detach()
        if finetuned_weight.is_meta:
            finetuned_weight = torch.zeros(finetuned_weight.size())
            print("A\n")
        finetuned_weight = finetuned_weight.to(device)

        compressed = BinaryDiff(
            base=base_weight,
            finetune=finetuned_weight,
        ).to(device)

        del submodule, base_weight
        setattr(module, subname, None)
        gc.collect()
        torch.cuda.empty_cache()
        setattr(module, subname, compressed)

    for name, module in finetuned_compressed_model.named_modules():
        for subname, submodule in module.named_children():
            if hasattr(submodule, "weight"):
                compress_module(name, subname, module, submodule, device)


# class CompressedModel(nn.Module):
#     def __init__(self, base_model, finetuned_model, device):
#         """
#         Precondition: base_model and finetuned_model are of the same architecture
#         """
#         super(CompressedModel, self).__init__()
#         # for each layer in both model
#         # compute the difference between them
#         for layer in base_model.state_dict().keys():
#             curr_base_weights = base_model.state_dict()[layer]
#             curr_ft_weights = finetuned_model.state_dict()[layer]

#             if curr_base_weights.is_meta:
#                 curr_base_weights = torch.zeros(curr_base_weights.size())
#             if curr_ft_weights.is_meta:
#                 curr_ft_weights = torch.zeros(curr_ft_weights.size())

#             base_weights = curr_base_weights.to(device)
#             finetuned_weights = curr_ft_weights.to(device)

#             # convert weights to positive/negative mask
#             # 1 if finetuned weight is greater than base weight
#             # -1 otherwise
#             diff = torch.where(finetuned_weights >= base_weights, 1.0, -1.0)
#             diff = diff.to(device)
#             # store the mask in the compressed model
#             self.register_parameter("mask", nn.Parameter(diff))
#             # also, add the trainable bitdelta to parameters
#             self.register_parameter("bitdelta", nn.Parameter(torch.tensor(0.5)))

#     def forward(self, x, base_model):
#         # for each layer in the model
#         for layer in self.state_dict().keys():
#             base_weights = base_model.state_dict()[layer]
#             mask = self.state_dict()["mask"]
#             bitdelta = self.state_dict()["bitdelta"]
#             # compute the compressed weights
#             compressed_weights = base_weights + bitdelta * mask
#             # apply the compressed weights to the input
#             x = torch.matmul(x, compressed_weights)
#         return x
