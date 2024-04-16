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
        mask = torch.where(diff > 0, 1, 0).to(base.device)

        self.register_buffer("mask", mask.T)
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
        del base, finetune, diff

    def forward(self, x):
        # repeated_mask = self.mask.unsqueeze(0).repeat(x.size(0), 1, 1)
        # print(x.size(), self.base.size(), self.coeff, self.mask.size())
        # print("\n")

        # convert datatype of x
        t_base = self.base.to(x.dtype)
        t_coeff = self.coeff.to(x.dtype)
        t_mask = self.mask.to(x.dtype)
        return x @ t_base + t_coeff * x @ t_mask

def compress_diff(base_model, finetuned_model, finetuned_compressed_model, device):
    def compress_module(parent_name, parent_module, name, module, device):
        # target_device = submodule.weight.device
        processed_name = parent_name.replace("distilbert.", '', 1)
        # print(f"Parent module: {processed_name}")
        # print(f"Compressing module: {name}")
        # print(f"{processed_name}.{name}")
        
        base_weight = base_model.get_submodule(f"{processed_name}.{name}").weight.detach().to(device)
        if base_weight.is_meta:
            base_weight = torch.zeros(base_weight.size())
        base_weight = base_weight.to(device)
        finetuned_weight = finetuned_model.get_submodule(f"{processed_name}.{name}").weight.detach().to(device)
        if finetuned_weight.is_meta:
            finetuned_weight = torch.zeros(finetuned_weight.size())
        finetuned_weight = finetuned_weight.to(device)

        compressed = BinaryDiff(
            base=base_weight,
            finetune=finetuned_weight,
        ).to(device)

        del module, base_weight
        setattr(parent_module, name, None)
        gc.collect()
        torch.cuda.empty_cache()
        setattr(parent_module, name, compressed)

    def recurse_and_replace(parent_name, parent_module):
        for name, module in list(parent_module.named_children()):
            if hasattr(module, "weight") and "lin" in name:
                # print(f"Compressing and replacing module: {name}")
                compress_module(parent_name, parent_module, name, module, device)

    for name, module in finetuned_compressed_model.named_modules():
        # print(f"Compressing and replacing module: {name}")
        recurse_and_replace(name, module)
            

def save_diff(finetuned_compressed_model, save_dir):
    diff_dict = {}

    for name, module in finetuned_compressed_model.named_modules():
        if isinstance(module, BinaryDiff):
            # diff_dict[name + ".mask"] = (module.mask == 1).bool().cpu()
            diff_dict[name + ".mask"] = module.mask
            diff_dict[name + ".coeff"] = module.coeff

    for name, param in finetuned_compressed_model.named_parameters():
        if param.requires_grad:
            diff_dict[name] = param

    torch.save(diff_dict, save_dir)


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
