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
from rgranet.data import get_dataset

class SLURM_Trainer():
    def __init__(self, config):
        self.config = config
    
    def __call__(self):
        init_dist_node()
        main.set_up_training(config=self.config)

def handle_slurm(config):
    slurm_config = config.distributed
    executor = submitit.AutoExecutor(folder=slurm_config.logs_folder, slurm_max_num_timeout=slurm_config.max_num_timeout)

    executor.update_parameters(
        mem_gb=12*slurm_config.ngpus,
        gpus_per_node=slurm_config.ngpus,
        tasks_per_node=slurm_config.ngpus,
        cpus_per_task=2,
        nodes=slurm_config.nnodes,
        timeout_min=5,
        slurm_partition=slurm_config.partition
    )

    coalesce(slurm_config.port, random.randint(49152, 65535))


    additional_parameters = {}

    if slurm_config.nodelist is not None:
        additional_parameters["nodelist"] = f"{slurm_config.nodelist}"
    
    if slurm_config.qos is not None:
        additional_parameters["qos"] = slurm_config.qos
    
    if slurm_config.account is not None:
        additional_parameters["account"] = slurm_config.account

    if slurm_config.mail_user is not None:
        additional_parameters["mail-type"] = slurm_config.mail_type
        additional_parameters["mail-user"] = slurm_config.mail_user
    
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

def init_dist_node(slurm_config):
    if "SLURM_JOB_ID" in os.environ:
        slurm_config.ngpus_per_node = torch.cuda.device_count()

        # requeue job on SLURM preemption
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)

        # find a common host name on all nodes
        cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
        stdout = subprocess.check_output(cmd.split())
        host_name = stdout.decode().splitlines()[0]
        slurm_config.dist_url = f'tcp://{host_name}:{slurm_config.port}'

        # distributed parameters
        slurm_config.rank = int(os.getenv('SLURM_NODEID')) * slurm_config.ngpus_per_node
        slurm_config.world_size = int(os.getenv('SLURM_NNODES')) * slurm_config.ngpus_per_node
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

def init_dist_gpu(slurm_config):
    job_env = submitit.JobEnvironment()
    slurm_config.logs_folder = Path(slurm_config.logs_folder) / str(job_env.job_id)
    slurm_config.gpu = job_env.local_rank
    slurm_config.rank = job_env.global_rank

    dist.init_process_group(backend="gloo", init_method=slurm_config.dist_url, world_size=slurm_config.world_size, rank=slurm_config.rank)
    torch.cuda.set_device(slurm_config.gpu)
    cudnn.benchmark = True
    slurm_config.main = (slurm_config.rank == 0)
    setup_for_distributed(slurm_config.main)







    
