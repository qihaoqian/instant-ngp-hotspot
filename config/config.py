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
    boundary_loss_weight : int = 350 # weight for boundary loss
    eikonal_loss_weight: int = 1 # weight for eikonal loss
    sign_loss_weight: int = 0 # weight for sign loss
    heat_loss_weight: int = 20 # weight for heat loss
    projection_loss_weight: int = 2 # weight for projection loss
    grad_direction_loss_weight: int = 5 # weight for 
    h: float = 1e-4 # step size for finite difference
    resolution: int = 512 # resolution for output mesh
    heat_loss_lambda: int = 4

@dataclass
class DataConfig(ConfigABC):
    dataset_path: str = 'data/armadillo.obj'
    train_size: int = 100
    valid_size: int = 1
    num_samples: int = 30000

@dataclass
class OptimizerConfig(ConfigABC):
    type: str = "Adam"               # optimizer type
    lr: float = 1e-4                 # learning rate
    weight_decay: float = 1e-6       # weight decay
    betas: Tuple[float, float] = (0.9, 0.999)  # betas parameter
    eps: float = 1e-15               # eps parameter
    
@dataclass
class SchedulerConfig(ConfigABC):
    type: str = "StepLR"  # scheduler type
    step_size: int = 10   # step size: adjust learning rate every 'step_size' epochs
    gamma: float = 1    # learning rate decay rate
    
@dataclass
class ModelConfig(ConfigABC):
    num_layers: int = 3
    hidden_dim: int = 64
    num_levels: int = 8
    base_resolution: int = 16
    desired_resolution: int = 2048
    sphere_radius: float = 1.6
    sphere_scale: float = 1.0
    use_sphere_post_processing: bool = False
    
@dataclass
class Config(ConfigABC):
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    seed: int = 0
    test: bool = False
    epochs: int = 50
    
