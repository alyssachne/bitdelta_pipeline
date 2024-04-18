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
        # note: use int8 to save memory
        mask = torch.where(diff > 0, 1, 0).to(base.device).to(torch.int8)

        self.register_buffer("mask", mask.T) # this is our bitdelta
        self.register_buffer("base", base.T)
        self.register_parameter(
            "coeff",
            nn.Parameter(
                torch.tensor(
                    quantile,
                    dtype=torch.float64,
                    requires_grad=False,
                    device=base.device,
                )
            ),
        )
        del base, finetune, diff

    def forward(self, x):
        repeated_mask = self.mask.unsqueeze(0).repeat(x.size(0), 1, 1)
        t = self.base.dtype

        # convert datatype of x
        t_base = self.base
        t_coeff = self.coeff.to(t)
        repeated_mask = repeated_mask.to(t)
        x = x.to(t)
        return x @ t_base + t_coeff * x @ repeated_mask

def compress_diff(base_model, finetuned_model, finetuned_compressed_model, device):
    def compress_module(parent_name, parent_module, name, module, device):
        # target_device = submodule.weight.device
        processed_name = parent_name.replace("distilbert.", '', 1)
        
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

    for parent_name, parent_module in finetuned_compressed_model.named_modules():
        for name, module in parent_module.named_children():
            if hasattr(module, "weight") and "lin" in name:
                # print(f"Compressing and replacing module: {name}")
                compress_module(parent_name, parent_module, name, module, device)
            

def save_diff(finetuned_compressed_model, save_dir):
    diff_dict = {}

    for name, module in finetuned_compressed_model.named_modules():
        if isinstance(module, BinaryDiff):
            diff_dict[name + ".mask"] = module.mask
            diff_dict[name + ".coeff"] = module.coeff

    for name, param in finetuned_compressed_model.named_parameters():
        if param.requires_grad:
            diff_dict[name] = param

    torch.save(diff_dict, save_dir)

@torch.no_grad()
def load_diff(model, diff_dir):
    device = model.device
    diff_dict = torch.load(diff_dir)

    for name, module in model.named_modules():
        if name + ".mask" in diff_dict:
            coeff = diff_dict[name + ".coeff"].to(device)
            mask = diff_dict[name + ".mask"].to(device)
            weight = mask * coeff

            module.weight.add_(weight.T.to(module.weight.dtype))
        elif name + ".weight" in diff_dict:
            module.weight = nn.Parameter(diff_dict[name + ".weight"].to(device).to(module.weight.dtype))

    return model