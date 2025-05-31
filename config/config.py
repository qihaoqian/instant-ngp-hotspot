from dataclasses import dataclass, field
from typing import Tuple
from config.config_abc import ConfigABC
from typing import Union

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
    eikonal_loss_surf_weight: int = 1 # weight for eikonal loss on surface
    eikonal_loss_space_weight: int = 3
    sign_loss_weight: int = 1 # weight for sign loss
    heat_loss_weight: int = 10 # weight for heat loss
    projection_loss_weight: int = 1 # weight for projection loss
    grad_direction_loss_weight: int = 20 # weight for gradient direction loss
    second_gradient_loss_weight: int = 0 # weight for second gradient loss
    h1: float = 1e-2 # step size for finite difference
    h2: float = 5e-2 # step size for second finite difference
    resolution: int = 2 # resolution for output mesh
    heat_loss_lambda: int = 4

@dataclass
class DataConfig(ConfigABC):
    dataset_path: str = 'data/armadillo.obj'
    train_size: int = 100
    valid_size: int = 1
    num_samples_surf: int = 20000
    num_samples_space: int = 10000

@dataclass
class OptimizerConfig(ConfigABC):
    type: str = "Adam"               # optimizer type
    lr: float = 1e-3                # learning rate
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
    encoding: str = "hashgrid"  # encoding type
    num_layers: int = 4
    hidden_dim: int = 128
    sphere_radius: float = 1.6
    sphere_scale: float = 1.0
    use_sphere_post_processing: bool = False
    
@dataclass
class HashGridConfig(ConfigABC):
    num_levels: int = 8
    base_resolution: int = 16
    desired_resolution: int = 2048
    
@dataclass    
class RegularGridConfig(ConfigABC):
    feature_dim: int = 32
    grid_dim: int = 3
    grid_min: tuple[float] = (-1, -1, -1)
    grid_max: tuple[float] = (1, 1, 1)
    grid_res: tuple[float] = (0.05,0.05,0.05)  # resolution of the grid
    
@dataclass
class Config(ConfigABC):
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    hash_grid: HashGridConfig = field(default_factory=HashGridConfig)
    reg_grid: RegularGridConfig = field(default_factory=RegularGridConfig)
    seed: int = 0
    test: bool = False
    epochs: int = 50
    
