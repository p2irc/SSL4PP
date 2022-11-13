import os
import subprocess

import torch
import torch.distributed as dist


def init_distributed_mode(launcher: str, backend: str) -> None:
    """
    Launches distributed training based on given launcher and backend.

    Args:
        launcher (str): {'pytorch', 'slurm'} Specifies if pytorch launch utitlity
            (torch.distributed.run) is being used or if running on a SLURM cluster.
        backend (str): {'nccl', 'gloo', 'mpi'} Specifies which backend to use
            when initializing a process group.
    """
    if launcher == "pytorch":
        launch_pytorch_dist(backend)
    elif launcher == "slurm":
        launch_slurm_dist(backend)
    else:
        raise RuntimeError(
            f"Invalid launcher type: {launcher}. Use 'pytorch' or 'slurm'."
        )


def launch_pytorch_dist(backend: str) -> None:
    """Initializes a distributed process group when using the pytorch
    distributed launch utility (torch.distributed.run).

    NOTE: This method relies on torch.distributed.run to set
    MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE and LOCAL_RANK
    as environment variables

    Args:
        backend (str): {'nccl', 'gloo', 'mpi'} Specifies which backend to use
            when initializing a process group.

    """
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, init_method="env://")
    disable_non_master_print()  # only print in master process
    torch.distributed.barrier()


def launch_slurm_dist(backend: str) -> None:
    """Initializes a distributed process group when process are spawned
    in a SLURM cluster (using the srun and sbatch commands).

    Args:
        backend (str): {'nccl', 'gloo', 'mpi'} Specifies which backend to use
            when initializing a process group.

    """
    # set the MASTER_ADDR, MASTER_PORT, RANK and WORLD_SIZE
    # as environment variables before initializing the process group
    if "MASTER_ADDR" not in os.environ:
        node_list = os.environ["SLURM_NODELIST"]
        os.environ["MASTER_ADDR"] = subprocess.getoutput(
            f"scontrol show hostname {node_list} | head -n1"
        )
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29400"
    os.environ["RANK"] = os.environ["SLURM_PROCID"]
    os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]

    local_rank = int(os.environ["SLURM_LOCALID"])
    torch.cuda.set_device(local_rank)
    print(f"Initializing distributed training in proces {local_rank}")
    dist.init_process_group(backend=backend, init_method="env://")
    disable_non_master_print()  # only print on master process
    torch.distributed.barrier()


# the following functions were adapted from:
# https://github.com/pytorch/vision/blob/main/references/classification/utils.py
def disable_non_master_print():
    """
    Disables printing if not master process. However, printing can be forced
    by adding a boolean flag, 'force', to the keyword arguments to the print
    function call.
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_main_process() or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized() -> bool:
    """
    Returns True if the PyTorch distribued package is available
    and a distributed process group has been initialized
    """
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    """
    Returns the total number of processes that have been initialized
    in a distributed process group. It returns 1 if the PyTorch distribued
    package is unavailable or the default process group has not been initialized.
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """
    Returns the global rank of the current process in the default distribued
    process group. Returns 0 if the PyTorch distribued package is unavailable
    or the default process group has not been initialized.
    """
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    """
    Checks if the current process in the default process group is the Master
    proces. The master process typically has a rank of 0.
    """
    return not is_dist_avail_and_initialized() or get_rank() == 0


# the following are from
# https://github.com/pytorch/vision/blob/main/references/detection/utils.py
def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.inference_mode():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
    return data_list
