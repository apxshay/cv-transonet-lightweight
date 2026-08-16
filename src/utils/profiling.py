import torch


def count_params(module, skip=()):
    n = 0
    for name, p in module.named_parameters():
        if name.split('.')[0] in skip:
            continue
        n += p.numel()
    return n


def gpu_mem():
    if not torch.cuda.is_available():
        return 0.0, 0.0
    return torch.cuda.max_memory_allocated() / 1e6, torch.cuda.max_memory_reserved() / 1e6
