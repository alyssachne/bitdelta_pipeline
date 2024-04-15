import torch

def pack(x, n_bits=32):
    """
    pack n_bits of x into a single integer
    
    x: bool tensor (*, K, N)
    return: int tensor (*, K // n_bits, N)
    """
    assert x.shape[-2] % n_bits == 0, "K must be divisible by n_bits"

    shift = torch.arange(n_bits, device=x.device)
    shape = x.shape[:-2]
    x = x.view(-1, x.shape[-2]//n_bits, n_bits, x.shape[-1])
    x = x << shift[None, None, :, None]
    x = x.sum(-2)
    x = x.view(*shape, *x.shape[-2:])
    
    # determine dtype
    if n_bits == 8:
        dtype = torch.uint8
    elif n_bits == 16:
        dtype = torch.int16
    elif n_bits == 32:
        dtype = torch.int32
    elif n_bits == 64:
        dtype = torch.int64

    return x.to(dtype)

def unpack(x, n_bits=32):
    """
    unpack n_bits of x into a single integer
    
    x: int tensor (*, K // n_bits, N)
    return: bool tensor (*, K, N)
    """
    shift = torch.arange(n_bits, device=x.device)
    shape = x.shape[:-2]
    x = x.view(-1, x.shape[-2], 1, x.shape[-1])
    x = (x >> shift[None, None, :, None]) & 0x1
    x = x.view(*shape, -1, x.shape[-1])
    return x.bool()
