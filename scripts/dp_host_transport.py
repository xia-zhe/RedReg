"""Synchronous host-staged transport for dense PyTorch1.8 DataParallel.

Retains the original replication/autograd functions, sharding and primary BN
buffer ownership. Only tensor movement and replica-gradient summation change.
All remote-device values pass through CPU; there is no NCCL or direct GPU peer
copy in these four communication functions. No compression or dtype conversion.
"""
import torch
from torch.cuda import comm
from torch.nn.parallel import comm as parallel_comm

_ORIGINALS = None


def index(device):
    if device is None:
        return torch.cuda.current_device()
    if isinstance(device, int):
        return device
    parsed = torch.device(device)
    if parsed.type == 'cpu':
        return -1
    return parsed.index if parsed.index is not None else torch.cuda.current_device()


def host_copy(tensor, destination):
    destination = index(destination)
    if tensor.is_cuda and tensor.get_device() == destination:
        return tensor
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)
    cpu = tensor.detach().to('cpu', non_blocking=False)
    if destination == -1:
        return cpu
    with torch.cuda.device(destination):
        return cpu.to(torch.device('cuda', destination), non_blocking=False)


@torch.no_grad()
def broadcast_coalesced(tensors, devices, buffer_size=10485760):
    tensors, devices = tuple(tensors), [index(x) for x in devices]
    assert devices and all(t.is_cuda and t.get_device() == devices[0] and not t.is_sparse for t in tensors)
    # In particular, return original tensors for device0: BN buffers must alias
    # the base module. Other replicas receive independent, same-dtype copies.
    return tuple(tuple(t if d == devices[0] else host_copy(t, d) for t in tensors) for d in devices)


@torch.no_grad()
def reduce_add(inputs, destination=None):
    inputs = tuple(inputs); destination = index(destination)
    assert inputs and all(t.is_cuda and not t.is_sparse and t.shape == inputs[0].shape and t.dtype == inputs[0].dtype for t in inputs)
    roots = [i for i, t in enumerate(inputs) if t.get_device() == destination]
    assert len(roots) == 1
    if len(inputs) == 1:
        return inputs[0]
    torch.cuda.synchronize(destination)
    result = inputs[roots[0]].clone()
    for i, tensor in enumerate(inputs):
        if i != roots[0]:
            result.add_(host_copy(tensor, destination))
    return result


@torch.no_grad()
def scatter(tensor, devices=None, chunk_sizes=None, dim=0, streams=None, *, out=None):
    if out is not None:
        raise NotImplementedError('Host DP transport does not use scatter(out=)')
    devices = [index(x) for x in devices]
    chunks = tensor.split(tuple(chunk_sizes), dim) if chunk_sizes is not None else tensor.chunk(len(devices), dim)
    return tuple(host_copy(chunk, d) for chunk, d in zip(chunks, devices))


@torch.no_grad()
def gather(tensors, dim=0, destination=None, *, out=None):
    tensors = tuple(tensors); destination = index(destination)
    assert tensors and all(t.is_cuda and not t.is_sparse for t in tensors)
    # Gather's existing autograd Function supplies the inverse scatter in its
    # backward, so these transport copies intentionally create no new graph.
    cpu = torch.cat([host_copy(t, -1) for t in tensors], dim=dim)
    result = host_copy(cpu, destination)
    if out is not None:
        out.copy_(result)
        return out
    return result


def install():
    global _ORIGINALS
    if _ORIGINALS is None:
        # In1.8 these are separate modules; DataParallel uses nn.parallel.comm.
        modules = list(dict.fromkeys([comm, parallel_comm]))
        _ORIGINALS = [(module, {name: getattr(module, name) for name in
                       ['broadcast_coalesced', 'reduce_add', 'scatter', 'gather']}) for module in modules]
    for module, _ in _ORIGINALS:
        module.broadcast_coalesced = broadcast_coalesced
        module.reduce_add = reduce_add
        module.scatter = scatter
        module.gather = gather


def restore():
    if _ORIGINALS is not None:
        for module, functions in _ORIGINALS:
            for name, function in functions.items():
                setattr(module, name, function)
