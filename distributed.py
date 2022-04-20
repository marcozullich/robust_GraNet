from types import SimpleNamespace
from setuptools import setup
import submitit
import os
import torch
import signal
import subprocess
import random
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from pprint import pprint
from pathlib import Path

import main
from rgranet.utils import coalesce


class SLURM_Trainer():
    def __init__(self, config):
        self.config = config
    
    def __call__(self):
        print(f"Starting distributed training with config: {self.config}")
        init_dist_node(self.config)
        main.set_up_training(config=self.config)

def handle_slurm(config):
    executor = submitit.AutoExecutor(folder=config.distributed.logs_folder, slurm_max_num_timeout=config.distributed.max_num_timeout)

    executor.update_parameters(
        mem_gb=12*config.distributed.ngpus,
        gpus_per_node=config.distributed.ngpus,
        tasks_per_node=config.distributed.ngpus,
        cpus_per_task=2,
        nodes=config.distributed.nnodes,
        timeout_min=5,
        slurm_partition=config.distributed.partition
    )

    config.distributed.port = coalesce(config.distributed.port, random.randint(49152, 65535))


    additional_parameters = {}

    if config.distributed.nodelist is not None:
        additional_parameters["nodelist"] = f"{config.distributed.nodelist}"
    
    if config.distributed.qos is not None:
        additional_parameters["qos"] = config.distributed.qos
    
    if config.distributed.account is not None:
        additional_parameters["account"] = config.distributed.account

    if config.distributed.mail_user is not None:
        additional_parameters["mail-type"] = config.distributed.mail_type
        additional_parameters["mail-user"] = config.distributed.mail_user
    
    executor.update_parameters(slurm_additional_parameters=additional_parameters)

    executor.update_parameters(name=config.net)

    pprint(executor.parameters)

    trainer = SLURM_Trainer(config)
    job = executor.submit(trainer)

def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()

def handle_sigterm(signum, frame):
    pass

def init_dist_node(config):
    if "SLURM_JOB_ID" in os.environ:
        config.distributed.ngpus_per_node = torch.cuda.device_count()

        # requeue job on SLURM preemption
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)

        # find a common host name on all nodes
        cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
        stdout = subprocess.check_output(cmd.split())
        host_name = stdout.decode().splitlines()[0]
        config.distributed.dist_url = f'tcp://{host_name}:{config.distributed.port}'

        # distributed parameters
        config.distributed.rank = int(os.getenv('SLURM_NODEID')) * config.distributed.ngpus_per_node
        config.distributed.world_size = int(os.getenv('SLURM_NNODES')) * config.distributed.ngpus_per_node
    else:
        raise RuntimeError("SLURM_JOB_ID not in environment. Cannot execute slurm.")

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def setup_distributed_debug_mode_config(config, job_env):
    ddm_config = {
        "jobno": job_env.job_id,
        "rank": job_env.global_rank,
    }
    config.distributed.debug_config = SimpleNamespace(**ddm_config)

def init_dist_gpu(config):
    job_env = submitit.JobEnvironment()
    config.distributed.logs_folder = Path(config.distributed.logs_folder) / str(job_env.job_id)
    config.distributed.gpu = job_env.local_rank
    config.distributed.rank = job_env.global_rank

    if hasattr(config.distributed, "debug") and config.distributed.debug:
        setup_distributed_debug_mode_config(config, job_env)
        print(f"Distributed debug mode: {config.distributed.debug_config}")

    dist.init_process_group(backend="gloo", init_method=config.distributed.dist_url, world_size=config.distributed.world_size, rank=config.distributed.rank)
    torch.cuda.set_device(config.distributed.gpu)
    cudnn.benchmark = True
    config.distributed.main = (config.distributed.rank == 0)
    setup_for_distributed(config.distributed.main)







    
