from dataclasses import dataclass, field
from typing import Tuple
from config.config_abc import ConfigABC

@dataclass
class TrainerConfig(ConfigABC):
    name : str = "ngp"  # name of the experiment
    eval_interval : int = 5 # interval for evaluation
    use_checkpoint : str = "latest" # checkpoint to use for evaluation
    workspace : str = "workspace" # workspace directory
    use_tensorboardX : bool = True # use tensorboard for logging
    max_keep_ckpt : int = 2 # maximum number of checkpoints to keep
    local_rank : int = 0 # local rank for distributed training
    world_size : int = 1 # world size for distributed training
    ema_decay : float = 0.95 # decay for exponential moving average
    boundary_loss_weight : float = 3e3 # weight for boundary loss
    eikonal_loss_weight: float = 5e1 # weight for eikonal loss
    sign_loss_weight: float = 1e2 # weight for sign loss
    heat_loss_weight: float = 5e1 # weight for heat loss
    h: float = 1e-4 # step size for finite difference
    resolution: int = 512 # resolution for output mesh
    heat_loss_lambda: int = 20

@dataclass
class DataConfig(ConfigABC):
    dataset_path: str = 'data/armadillo.obj'
    train_size: int = 100
    valid_size: int = 1
    num_samples: int = 2**18

@dataclass
class OptimizerConfig(ConfigABC):
    type: str = "Adam"               # optimizer type
    lr: float = 1e-3                 # learning rate
    weight_decay: float = 1e-6       # weight decay
    betas: Tuple[float, float] = (0.9, 0.99)  # betas parameter
    eps: float = 1e-15               # eps parameter
    
@dataclass
class SchedulerConfig(ConfigABC):
    type: str = "StepLR"  # scheduler type
    step_size: int = 10   # step size: adjust learning rate every 'step_size' epochs
    gamma: float = 0.1    # learning rate decay rate
    
@dataclass
class Config(ConfigABC):
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    seed: int = 0
    test: bool = False
    epochs: int = 20