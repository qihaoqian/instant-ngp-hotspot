import os
import glob
import tqdm
import random
import tensorboardX

import numpy as np

import time

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist

import trimesh
import mcubes
from rich.console import Console
from torch_ema import ExponentialMovingAverage

import packaging
import io
from PIL import Image
import tarfile
import os

def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if packaging.version.parse(torch.__version__) < packaging.version.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True


def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy() # [N, 1] --> [x, y, z]
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u

def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    #print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    #print(u.shape, u.max(), u.min(), np.percentile(u, 50))
    
    vertices, triangles = mcubes.marching_cubes(u, threshold)

    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


class Trainer(object):
    def __init__(self, 
                 name, # name of this experiment
                 model, # network 
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 optimizer=None, # optimizer
                 ema_decay=None, # if use EMA, set the decay
                 lr_scheduler=None, # scheduler
                 max_epochs=None, # max epochs
                 metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 fp16=False, # amp optimize level
                 eval_interval=1, # eval once every $ epoch
                 max_keep_ckpt=2, # max num of saved ckpts in disk
                 workspace='workspace', # workspace to save logs & ckpts
                 best_mode='min', # the smaller/larger result, the better
                 use_loss_as_metric=True, # use loss as the first metirc
                 report_metric_at_train=False, # also report metrics at training
                 use_checkpoint="latest", # which ckpt to use at init time
                 use_tensorboardX=True, # whether to use tensorboard for logging
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
                 mape_loss_weight=0, # weight for mape loss
                 boundary_loss_weight=3e3, # weight for boundary loss
                 eikonal_loss_surf_weight=1, # weight for eikonal loss on surface
                 eikonal_loss_space_weight=3, # weight for eikonal loss on space
                 sign_loss_free_weight=1, # weight for sign loss for free points
                 sign_loss_occ_weight=1, # weight for sign loss for occupied points
                 heat_loss_weight=5e1, # weight for heat loss
                 projection_loss_weight=0, # weight for projection loss
                 grad_dir_loss_weight=0, # weight for gradient direction loss
                 sec_grad_loss_weight=0, # weight for second gradient loss
                 h1=1e-2, # step size for finite difference
                 h2=1e-1, # step size for second finite difference
                 heat_loss_lambda=None # lambda for heat loss
                 ):
        
        self.name = name
        self.mute = mute
        self.max_epochs = max_epochs
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()
        self.mape_loss_weight = mape_loss_weight
        self.boundary_loss_weight = boundary_loss_weight
        self.eikonal_loss_surf_weight = eikonal_loss_surf_weight
        self.eikonal_loss_space_weight = eikonal_loss_space_weight
        self.heat_loss_weight = heat_loss_weight
        self.sign_loss_free_weight = sign_loss_free_weight
        self.sign_loss_occ_weight = sign_loss_occ_weight
        self.projection_loss_weight = projection_loss_weight
        self.grad_dir_loss_weight = grad_dir_loss_weight
        self.sec_grad_loss_weight = sec_grad_loss_weight
        self.h1 = h1
        self.h2 = h2
        self.heat_loss_lambda = heat_loss_lambda
        self.proj_loss_switch = False

        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4) # naive adam
        else:
            self.optimizer = optimizer(self.model)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1) # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [], # metrics[0], or valid_loss
            "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
            }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)        
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth.tar"
            os.makedirs(self.ckpt_path, exist_ok=True)
            
        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else: # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

    def __del__(self):
        if self.log_ptr: 
            self.log_ptr.close()

    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute: 
                #print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr: 
                print(*args, file=self.log_ptr)
                self.log_ptr.flush() # write immediately to file
                
    # def plot_space_gradient(self, X_surf, X_space, grad_space, title=None):
    #     from mpl_toolkits.mplot3d import Axes3D

    #     # Convert torch tensors to numpy arrays
    #     X_np = X_space.detach().cpu().numpy()
    #     grad_np = grad_space.detach().cpu().numpy()

    #     # Subsample for clarity
    #     num_vectors = min(200, X_np.shape[0])
    #     idx = np.random.choice(X_np.shape[0], num_vectors, replace=False)
    #     X_sub = X_np[idx]
    #     grad_sub = grad_np[idx]

    #     # Create 3D quiver plot
    #     fig = plt.figure(figsize=(8, 6))
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.quiver(
    #         X_sub[:, 0], X_sub[:, 1], X_sub[:, 2],
    #         grad_sub[:, 0], grad_sub[:, 1], grad_sub[:, 2],
    #         length=0.1, normalize=True
    #     )

    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     ax.set_zlabel('Z')
    #     ax.set_title('Gradient Directions Visualization')

    #     plt.savefig(f'gradient.png', dpi=300)
    #     plt.show(block=True)
        

    def plot_interactive_space_gradient(
        self,
        X_surf, X_space, grad_space,
        title: str = "Interactive Gradient Visualization",
        n_surf: int = 300,   # 要采样的表面点数量
        n_vec: int = 200,    # 要采样的梯度向量数量
        vec_scale: float = 0.2  # 锥体长度缩放因子
    ):
        import plotly.graph_objects as go
        """
        X_surf:      torch.Tensor of shape (N_surf, 3)
        X_space:     torch.Tensor of shape (N_space, 3)
        grad_space:  torch.Tensor of shape (N_space, 3)
        """

        # 转为 NumPy
        surf_np = X_surf.detach().cpu().numpy()
        space_np = X_space.detach().cpu().numpy()
        grad_np  = grad_space.detach().cpu().numpy()

        # 随机采样表面点
        m = min(n_surf, surf_np.shape[0])
        idx_s = np.random.choice(surf_np.shape[0], m, replace=False)
        surf_pts = surf_np[idx_s]

        # 随机采样梯度向量
        k = min(n_vec, space_np.shape[0])
        idx_v = np.random.choice(space_np.shape[0], k, replace=False)
        vec_pts  = space_np[idx_v]
        vec_dirs = grad_np[idx_v]

        # 构造 Plotly trace：表面点
        scatter = go.Scatter3d(
            x=surf_pts[:,0], y=surf_pts[:,1], z=surf_pts[:,2],
            mode='markers',
            marker=dict(size=2, color='royalblue'),
            name='Surface Points'
        )
        # 构造 Plotly trace：梯度锥体
        cones = go.Cone(
            x=vec_pts[:,0], y=vec_pts[:,1], z=vec_pts[:,2],
            u=vec_dirs[:,0], v=vec_dirs[:,1], w=vec_dirs[:,2],
            sizemode="scaled", 
            sizeref=vec_scale,      # 试不同值看效果，越大锥体越粗越长
            anchor="tail",
            showscale=False,
            name='Gradient Vectors'
        )

        fig = go.Figure(data=[scatter, cones])
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis=dict(
                    title='X',
                    range=[-1, 1],
                    autorange=False
                ),
                yaxis=dict(
                    title='Y',
                    range=[-1, 1],
                    autorange=False
                ),
                zaxis=dict(
                    title='Z',
                    range=[-1, 1],
                    autorange=False
                ),
                aspectmode='cube'  # 保证三个轴等比例
            ),
            margin=dict(l=0, r=0, t=40, b=0)
        )
        #fig.show()
        grad_visual_dir = os.path.join(self.workspace, "grad_visual")
        os.makedirs(grad_visual_dir, exist_ok=True)
        ws_name = os.path.basename(self.workspace)
        filename = f"{ws_name}_{title}.html"
        fig.write_html(
            os.path.join(grad_visual_dir, filename),
            include_plotlyjs="cdn",
            auto_open=False
        )
        
    ### ------------------------------	
    def train_step(self, data):
        # assert batch_size == 1
        X_surf = data["points_surf"][0] # [B, 3]
        y_surf = data["sdfs_surf"][0] # [B]
        X_occ = data["points_occupied"][0] # [B, 3], inside the object
        y_occ = data["sdfs_occupied"][0] # [B]
        X_free = data["points_free"][0] # [B, 3], outside the object
        y_free = data["sdfs_free"][0] # [B]
        # dists = data['points_space_dists'][0]

        X = torch.cat([X_surf, X_occ, X_free], dim=0)
        y = torch.cat([y_surf, y_occ, y_free], dim=0)
        
        X_space = X[X_surf.shape[0]:]
        # X.requires_grad_(True)
        
        y_pred = self.model(X)
        surf_pred = y_pred[:X_surf.shape[0]]
        free_pred = y_pred[X_surf.shape[0]+X_occ.shape[0]:]
        occ_pred = y_pred[X_surf.shape[0]:X_surf.shape[0]+X_occ.shape[0]]
        space_pred = y_pred[X_surf.shape[0]:]

        def finite_diff_grad(model, X, h1=1e-4):
            # X: [B, 3] Assume input is 3D coordinates
            grads = []
            valid_min = torch.tensor([-1,-1,-1], device=X.device) + h1
            valid_max = torch.tensor([1,1,1], device=X.device) - h1
            mask = ((X > valid_min) & (X < valid_max)).all(dim=1)
            idx = torch.nonzero(mask, as_tuple=False).squeeze(1)  # 返回满足条件的行索引
            X_safe = X[idx]
            for i in range(X_safe.shape[1]):
                # Create a zero tensor with the same shape as X
                offset = torch.zeros_like(X_safe)
                offset[:, i] = h1  # Only increase the i-th dimension by a small offset
                # Calculate SDF values with positive and negative perturbations
                sdf_plus = model(X_safe + offset)
                sdf_minus = model(X_safe - offset)
                # Calculate the gradient in the i-th direction using central difference formula
                grad_i = (sdf_plus - sdf_minus) / (2 * h1)
                grads.append(grad_i)  # Adjust the dimension for later concatenation
            # Concatenate the gradients in each direction to get a [B, 3] gradient tensor
            grad = torch.cat(grads, dim=-1)
            return grad , idx
        
        def second_derivative_loss(model, X, h1=1e-2, h2=1e-1):
            """
            Compute the second derivative loss for the model at points X.
            This is a placeholder function and should be implemented based on the specific requirements.
            """
            grads_1 = []
            grads_2 = []
            valid_min = torch.tensor([-1,-1,-1], device=X.device) + h1 + h2
            valid_max = torch.tensor([1,1,1], device=X.device) - h1 - h2
            X_safe = X[(X > valid_min).all(dim=1) & (X < valid_max).all(dim=1)]
            for i in range(X_safe.shape[1]):
                # Create a zero tensor with the same shape as X_safe
                offset1 = torch.zeros_like(X_safe)
                offset2 = torch.zeros_like(X_safe)
                offset1[:, i] = h1
                offset2[:, i] = h2
                grad_1 = (model(X_safe + offset2 + offset1) - model(X_safe + offset2 - offset1)) / (2 * h1)
                grad_2 = (model(X_safe - offset2 + offset1) - model(X_safe - offset2 - offset1)) / (2 * h1)
                grads_1.append(grad_1)
                grads_2.append(grad_2)
                # sec_grad_i = (model(X+offset1+offset2)-model(X+offset1-offset2) - model(X-offset1+offset2) + model(X-offset1-offset2)) / (4 * h1 * h2)
            grads_1 = torch.cat(grads_1, dim=-1)  # [B, 3]
            grads_2 = torch.cat(grads_2, dim=-1)  # [B, 3]
            
            sec_grads = (grads_1 - grads_2) / (2 * h2)  # [B, 3]
            # 直接在计算 loss 时做条件筛选和乘法，避免显式创建中间 tensor
            sec_grad_loss = ((sec_grads.abs() >= 0.5).float() * sec_grads).norm(dim=-1).sum() / sec_grads.shape[0]

            # dot_prod = (grads_1 * grads_2).sum(dim=-1, keepdim=True)  
            # sec_scalar = (grads_1.norm(dim=-1, keepdim=True) - grads_2.norm(dim=-1, keepdim=True)) / (2 * h2)   
            # sec_grads = torch.where(
            #     dot_prod < 0,                     # 条件
            #     grads_1 + grads_2,                # True 分支 → 向量和
            #     sec_scalar.expand_as(grads_1)     # False 分支 → 标量结果按通道广播
            # )
            return sec_grad_loss
        
                
        # # Compute gradients using finite difference
        # if not (self.eikonal_loss_space_weight == 0 and self.heat_loss_weight == 0 and self.projection_loss_weight == 0):
        #     # # Calculate gradient using finite difference method
        #     gradients = finite_diff_grad(self.model, X, h1=self.h1)
        #     grad_space = gradients[X_surf.shape[0]:]
        #     # # use autograd
        #     # gradients = torch.autograd.grad(
        #     #     y_pred, X,
        #     #     grad_outputs=torch.ones_like(y_pred),
        #     #     create_graph=True, retain_graph=True, only_inputs=True
        #     # )[0]
        
        def heat_loss(preds, grads=None, sample_pdfs=None, heat_lambda=4):
            heat = torch.exp(-heat_lambda * preds.abs())
            loss = 0.5 * heat ** 2 * (grads.norm(dim=1) ** 2 + 1)

            if sample_pdfs is not None:
                sample_pdfs = sample_pdfs.squeeze(-1)
                loss = loss.squeeze(-1)
                loss /= sample_pdfs
            loss = loss.mean()

            return loss
        
        # mape loss
        if self.mape_loss_weight != 0:
            difference = (surf_pred-y_surf).abs()
            scale = 1 / (y_surf.abs() + 1e-2)
            loss_mape = difference * scale
            loss_mape = loss_mape.mean()
        else:
            loss_mape = torch.tensor(0.0, device=y_pred.device)
            
        # boundary loss
        if self.boundary_loss_weight != 0:
            loss_boundary = (surf_pred-y_surf).abs().mean()
        else:
            loss_boundary = torch.tensor(0.0, device=y_pred.device)
        
        # sign loss
        if self.sign_loss_free_weight != 0:
            loss_sign_free = (torch.tanh(100 * free_pred) - 1).abs().mean()
            # loss_sign_free = torch.exp(-1e2 * free_pred).mean()
        else:
            loss_sign_free = torch.tensor(0.0, device=y_pred.device)
        
        if self.sign_loss_occ_weight != 0:
            loss_sign_occ = (torch.tanh(100 * occ_pred) + 1).abs().mean()
            # loss_sign_occ = torch.exp(1e2 * occ_pred).mean()
        else:
            loss_sign_occ = torch.tensor(0.0, device=y_pred.device)
        
        # Eikonal loss
        if self.eikonal_loss_surf_weight != 0 and self.eikonal_loss_space_weight != 0:
            grad_surf, grad_surf_idx = finite_diff_grad(self.model, X_surf, h1=self.h1)
            X_surf_safe = X_surf[grad_surf_idx]
            grad_occ, grad_occ_idx = finite_diff_grad(self.model, X_occ, h1=self.h1)
            X_occ_safe = X_occ[grad_occ_idx]
            grad_free, grad_free_idx = finite_diff_grad(self.model, X_free, h1=self.h1)
            X_free_safe = X_free[grad_free_idx]
            loss_eikonal_surf = (grad_surf.norm(dim=1) - 1).abs().mean()
            grad_space = torch.cat([grad_occ, grad_free], dim=0)  # [B, 3]
            grad_space_idx = torch.cat([grad_occ_idx, grad_free_idx], dim=0)  # [B]
            X_space_safe = torch.cat([X_occ_safe, X_free_safe], dim=0)
            loss_eikonal_space = (grad_space.norm(dim=1) - 1).abs().mean()
        else:
            loss_eikonal_surf = torch.tensor(0.0, device=y_pred.device)
            loss_eikonal_space = torch.tensor(0.0, device=y_pred.device)
        
        # Heat loss
        if self.heat_loss_weight != 0:
            if 'grad_space' not in locals():
                X_space = torch.cat([X_occ, X_free], dim=0)
                grad_space, grad_space_idx = finite_diff_grad(self.model, X_space, h1=self.h1)
            loss_heat = heat_loss(preds=space_pred[grad_space_idx], grads=grad_space, sample_pdfs=None, heat_lambda=self.heat_loss_lambda)
        else:
            loss_heat = torch.tensor(0.0, device=y_pred.device)
        
        if self.grad_dir_loss_weight != 0 or self.projection_loss_weight != 0:
            # 计算每个投影点到全部 surface 点的欧氏距离，找到最小的surf点
            d2_space = torch.cdist(X_space_safe, X_surf_safe)        # (X_free, |surf|)
            nn_idx_space = d2_space.argmin(dim=1)    
            # 构造 ground-truth 方向向量
            X_nn_space = X_surf_safe[nn_idx_space]                   # (N_proj,3)
            dir_gt_space = X_space_safe - X_nn_space              # (N_proj,3)
            # 归一化
            dir_gt_space_norm = dir_gt_space / (dir_gt_space.norm(dim=1, keepdim=True) + 1e-8)
            # dist_gt_space = dir_gt_space.norm(dim=1)  # (N_proj,)
            
        # gradient direction loss
        if self.grad_dir_loss_weight != 0:
            # X_free_safe = X_free[grad_free_idx]
            # d2_free = torch.cdist(X_free_safe, X_surf)        # (X_free, |surf|)
            # nn_idx_free = d2_free.argmin(dim=1) 
            # X_nn_free = X_surf[nn_idx_free]                   # (N_proj,3)
            
            # dir_gt_free = X_free_safe - X_nn_free              # (N_proj,3)
            # dir_gt_free_norm = dir_gt_free / (dir_gt_free.norm(dim=1, keepdim=True) + 1e-8)
            
            # if self.epoch == 1 and self.local_step == 1:
            #         self.plot_interactive_space_gradient(
            #             X_surf, X_space_safe, dir_gt_space_norm,
            #             title=f"Ground Truth Space Gradient",
            #             n_surf=3000,
            #             n_vec=200,
            #             vec_scale=0.2
            #         )                  
            # if self.epoch % self.eval_interval == 0 and self.local_step == 1:
            #     self.plot_interactive_space_gradient(
            #         X_surf, X_space_safe, grad_space,
            #         title=f"Space Gradient epoch_{self.epoch}",
            #         n_surf=3000,
            #         n_vec=200,
            #         vec_scale=0.2
            #     )
            #     print("\nAverage free point gradients norm:", grad_free.norm(dim=1).mean().item())    
            # # 计算梯度方向与 ground-truth 方向的点积
            # loss_grad_dir = (1.0 - dir_gt_free_norm * (grad_free / (grad_free.norm(dim=1, keepdim=True).detach() + 1e-8))).mean()
            
            grad_nn_space = grad_surf[nn_idx_space]
            # 计算梯度方向与 ground-truth 方向的点积
            grad_space_norm = grad_space / (grad_space.norm(dim=1, keepdim=True) + 1e-8)
            grad_nn_space_norm = grad_nn_space / (grad_nn_space.norm(dim=1, keepdim=True) + 1e-8)
            loss_grad_dir = (1.0 - (grad_space_norm * grad_nn_space_norm).sum(dim=1)).mean()

        else:
            loss_grad_dir = torch.tensor(0.0, device=y_pred.device)
            
        # Projection loss
        if self.projection_loss_weight != 0:
            X_proj = X_space_safe - dir_gt_space_norm * space_pred[grad_space_idx].abs()
            
            # 计算每个投影点到全部 surface 点的欧氏距离，并取最小
            #    torch.cdist 返回 (N_proj, |surf|) 的距离矩阵
            dists = torch.cdist(X_proj, X_surf).min(dim=1)[0]   # (N_proj,) loss 不下降

            # loss_projection = (dists*dist_gt_space).abs().mean()
            loss_projection = dists.abs().mean()
        else:
            loss_projection = torch.tensor(0.0, device=y_pred.device)
        
        if self.sec_grad_loss_weight != 0:
            # 计算二阶导数损失
            loss_sec_grad = second_derivative_loss(self.model, X_space, h1=self.h1, h2=self.h2)  # [B, 3]
        else:
            loss_sec_grad = torch.tensor(0.0, device=y_pred.device)
            
        def check_loss(name, value):
            if torch.isnan(value):
                raise RuntimeError(f"NaN detected in {name}")
            return value
        
        # Compute the weighted losses
        weighted_loss_mape = loss_mape * self.mape_loss_weight
        weighted_loss_boundary = loss_boundary * self.boundary_loss_weight
        weighted_loss_eikonal_surf = loss_eikonal_surf * self.eikonal_loss_surf_weight
        weighted_loss_eikonal_space = loss_eikonal_space * self.eikonal_loss_space_weight
        weighted_loss_sign_free = loss_sign_free * self.sign_loss_free_weight
        weighted_loss_sign_occ = loss_sign_occ * self.sign_loss_occ_weight
        weighted_loss_heat = loss_heat * self.heat_loss_weight
        weighted_loss_projection = loss_projection * self.projection_loss_weight
        weighted_loss_grad_dir = loss_grad_dir *  self.grad_dir_loss_weight
        weighted_loss_sec_grad = loss_sec_grad * self.sec_grad_loss_weight

        # Sum up the total loss
        loss = (weighted_loss_mape + weighted_loss_boundary + weighted_loss_eikonal_surf + weighted_loss_eikonal_space
        + weighted_loss_sign_free+ loss_sign_occ + weighted_loss_heat + weighted_loss_projection + weighted_loss_grad_dir + weighted_loss_sec_grad)

        return y_pred, y, loss, loss_mape, loss_boundary, loss_eikonal_surf, loss_eikonal_space, loss_sign_free, loss_sign_occ, loss_heat, loss_projection, loss_grad_dir, loss_sec_grad

    def eval_step(self, data):
        return self.train_step(data)

    def test_step(self, data):  
        X_surf = data["points_surf"][0] # [B, 3]
        y_surf = data["sdfs_surf"][0] # [B]
        X_occ = data["points_occupied"][0] # [B, 3], inside the object
        y_occ = data["sdfs_occupied"][0] # [B]
        X_free = data["points_free"][0] # [B, 3], outside the object
        y_free = data["sdfs_free"][0] # [B]

        X = torch.cat([X_surf, X_occ, X_free], dim=0)
        y = torch.cat([y_surf, y_occ, y_free], dim=0)
        y_pred = self.model(X)
        
        # X_space = X[X_surf.shape[0]:]
        # space_pred = y_pred[X_surf.shape[0]:]
        # grad_space = self.finite_diff_grad(self.model, X_space, h=self.h)
        
        return y_pred        

    def save_mesh(self, save_path=None, resolution=256):
        if save_path is None:
            save_path = os.path.join(self.workspace, 'validation', f'{self.name}_{self.epoch}.ply')

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        def query_func(pts):
            pts = pts.to(self.device)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    sdfs = self.model(pts)
            return sdfs

        bounds_min = torch.FloatTensor([-1, -1, -1])
        bounds_max = torch.FloatTensor([1, 1, 1])
        
        # 提取SDF值
        u = extract_fields(bounds_min, bounds_max, resolution, query_func)
        
        # 每次调用都保存xy截面到tensorboard
        z_index = u.shape[2] // 2
        cross_section = u[:, :, z_index]
        extent = [bounds_min[0], bounds_max[0], bounds_min[1], bounds_max[1]]
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cross_section.T, origin='lower', extent=extent, cmap='jet')
        plt.colorbar(label='SDF Distance')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'SDFs xy Cross Section (z = 0 at epoch{self.epoch})')
        
        # 保存到tensorboard
        if self.use_tensorboardX and self.local_rank == 0:
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            image = Image.open(buf)
            image = np.array(image)
            self.writer.add_image(f'sdfs_cross_section_xy/epoch{self.epoch}', image, self.epoch, dataformats='HWC')
            buf.close()
        
        # 在最后一个epoch时，额外保存三个方向的截面到results文件夹
        if self.epoch == self.max_epochs:
            self.log(f"==> Saving SDF cross sections at final epoch {self.epoch}")
            results_dir = os.path.join(self.workspace, 'results')
            os.makedirs(results_dir, exist_ok=True)
            
            # 创建2x3的子图布局，与dataset.py的plot_all_sdf_combined保持一致
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # 使用自定义颜色映射用于二值化图
            from matplotlib.colors import ListedColormap
            colors = ['blue', 'white', 'red']  # 内部、表面、外部
            binary_cmap = ListedColormap(colors)
            
            # 定义切面数据和对应的坐标范围
            v0, v1 = (-1.0, 1.0)
            
            # x轴切面：y-z平面 (x=0)
            x_index = u.shape[0] // 2
            x_slice = u[x_index, :, :]  # shape: (y_dim, z_dim)
            
            # y轴切面：x-z平面 (y=0) 
            y_index = u.shape[1] // 2
            y_slice = u[:, y_index, :]  # shape: (x_dim, z_dim)
            
            # z轴切面：x-y平面 (z=0)
            z_index = u.shape[2] // 2
            z_slice = u[:, :, z_index]  # shape: (x_dim, y_dim)
            
            slices = [x_slice, y_slice, z_slice]
            axes_names = ['x', 'y', 'z']
            
            for i, (axis, slice_data) in enumerate(zip(axes_names, slices)):
                # 确定剩下的两个维度对应的标签，与dataset.py保持一致
                dims = ['x', 'y', 'z']
                dims.remove(axis)
                xlabel, ylabel = dims
                
                # 直接转置所有切面数据以匹配GT的显示方式
                display_data = slice_data.T
                
                # 第一行：普通SDF切面图
                im1 = axes[0, i].imshow(
                    display_data,
                    origin='lower',
                    extent=[v0, v1, v0, v1],
                    cmap='jet',
                    vmin=np.min(display_data),
                    vmax=np.max(display_data)
                )
                plt.colorbar(im1, ax=axes[0, i], label='SDF value')
                axes[0, i].set_xlabel(xlabel)
                axes[0, i].set_ylabel(ylabel)
                axes[0, i].set_title(f"Pred SDF slice on {axis}={0.0:.2f}")
                
                # 第二行：二值化SDF切面图
                binary_slice = np.zeros_like(display_data)
                binary_slice[display_data > 0] = 1   # 外部区域
                binary_slice[display_data < 0] = -1  # 内部区域
                
                im2 = axes[1, i].imshow(
                    binary_slice,
                    origin='lower',
                    extent=[v0, v1, v0, v1],
                    cmap=binary_cmap,
                    vmin=-1,
                    vmax=1
                )
                
                # 只在最后一个子图添加颜色条
                if i == 2:
                    cbar = plt.colorbar(im2, ax=axes[1, i], ticks=[-1, 0, 1])
                    cbar.set_ticklabels(['Inside (SDF < 0)', 'Surface (SDF ≈ 0)', 'Outside (SDF > 0)'])
                
                # 绘制SDF=0的等值线
                contour = axes[1, i].contour(
                    np.linspace(v0, v1, display_data.shape[0]),
                    np.linspace(v0, v1, display_data.shape[1]),
                    display_data,
                    levels=[0],
                    colors=['black'],
                    linewidths=2
                )
                axes[1, i].clabel(contour, inline=True, fontsize=8, fmt='Surface')
                
                axes[1, i].set_xlabel(xlabel)
                axes[1, i].set_ylabel(ylabel)
                axes[1, i].set_title(f"SDF Binary Regions Slice {axis}={0.0:.2f}")
            
            # 调整布局并保存
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'sdf_pred.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        plt.close()
        
        vertices, triangles = extract_geometry(bounds_min, bounds_max, resolution=resolution, threshold=0, query_func=query_func)
        print(f"==> vertices: {vertices.shape}, triangles: {triangles.shape}")
        # if triangles.shape[0] == 0 or triangles.shape[0] > 1100000:
        #     self.log(f"==> No valid mesh extracted, skipping save.")
        #     return

        mesh = trimesh.Trimesh(vertices, triangles, process=False) # important, process=True leads to seg fault...
        mesh.export(save_path)

        self.log(f"==> Finished saving mesh.")

    ### ------------------------------

    def train(self, train_loader, valid_loader, max_epochs):
        def _backup_code(self):
            package_dir = os.path.dirname(os.path.abspath(__file__))
            with tarfile.open(os.path.join(self.workspace, "code.tar.gz"), "w:gz") as file:
                file.add(package_dir, arcname=os.path.basename(package_dir))
        _backup_code(self)
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))
            
        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch

            self.train_one_epoch(train_loader)

            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.eval_interval == 0:
                # self.evaluate_one_epoch(valid_loader)
                self.save_mesh()
                self.save_checkpoint(full=False, best=True)

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader):
        #if os.path.exists(self.best_path):
        #    self.load_checkpoint(self.best_path)
        #else:
        #    self.load_checkpoint()
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader)
        self.use_tensorboardX = use_tensorboardX



    def prepare_data(self, data):
        if isinstance(data, list):
            for i, v in enumerate(data):
                if isinstance(v, np.ndarray):
                    data[i] = torch.from_numpy(v).to(self.device, non_blocking=True)
                if torch.is_tensor(v):
                    data[i] = v.to(self.device, non_blocking=True)
        elif isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    data[k] = torch.from_numpy(v).to(self.device, non_blocking=True)
                if torch.is_tensor(v):
                    data[k] = v.to(self.device, non_blocking=True)
        elif isinstance(data, np.ndarray):
            data = torch.from_numpy(data).to(self.device, non_blocking=True)
        else: # is_tensor, or other similar objects that has `to`
            data = data.to(self.device, non_blocking=True)

        return data

    def train_one_epoch(self, loader):
        self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)
        
        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data in loader:
            
            self.local_step += 1
            self.global_step += 1
            
            data = self.prepare_data(data)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, loss, loss_mape, loss_boundary, loss_eikonal_surf, loss_eikonal_space, loss_sign_free, loss_sign_occ, loss_heat, loss_projection, loss_grad_dir, loss_sec_grad = self.train_step(data)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.ema is not None:
                self.ema.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.local_rank == 0:
                if self.report_metric_at_train:
                    for metric in self.metrics:
                        metric.update(preds, truths)
                        
                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)
                    self.writer.add_scalar("train/loss_mape", loss_mape.item(), self.global_step)
                    self.writer.add_scalar("train/loss_boundary", loss_boundary.item(), self.global_step)
                    self.writer.add_scalar("train/loss_eikonal_surf", loss_eikonal_surf.item(), self.global_step)
                    self.writer.add_scalar("train/loss_eikonal_space", loss_eikonal_space.item(), self.global_step)
                    self.writer.add_scalar("train/loss_sign_free", loss_sign_free.item(), self.global_step)
                    self.writer.add_scalar("train/loss_sign_occ", loss_sign_occ.item(), self.global_step)
                    self.writer.add_scalar("train/loss_heat", loss_heat.item(), self.global_step)
                    self.writer.add_scalar("train/loss_projection", loss_projection.item(), self.global_step)
                    self.writer.add_scalar("train/loss_grad_dir", loss_grad_dir.item(), self.global_step)
                    self.writer.add_scalar("train/loss_sec_grad", loss_sec_grad.item(), self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                pbar.update(loader.batch_size)

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")


    def evaluate_one_epoch(self, loader):
        self.log(f"++> Evaluate at epoch {self.epoch} ...")

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0
            for data in loader:    
                self.local_step += 1
                
                data = self.prepare_data(data)

                if self.ema is not None:
                    self.ema.store()
                    self.ema.copy_to()
            
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, truths, loss, loss_mape, loss_boundary, loss_eikonal, loss_sign, loss_heat = self.eval_step(data)

                if self.ema is not None:
                    self.ema.restore()
                
                # all_gather/reduce the statistics (NCCL only support all_*)
                if self.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / self.world_size
                    
                    preds_list = [torch.zeros_like(preds).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_list, preds)
                    preds = torch.cat(preds_list, dim=0)

                    truths_list = [torch.zeros_like(truths).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(truths_list, truths)
                    truths = torch.cat(truths_list, dim=0)

                loss_val = loss.item()
                total_loss += loss_val

                # only rank = 0 will perform evaluation.
                if self.local_rank == 0:

                    for metric in self.metrics:
                        metric.update(preds, truths)

                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                    pbar.update(loader.batch_size)

        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(result if self.best_mode == 'min' else - result) # if max mode, use -result
            else:
                self.stats["results"].append(average_loss) # if no metric, choose best by min loss

            for metric in self.metrics:
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    def save_checkpoint(self, full=False, best=False):

        state = {
            'epoch': self.epoch,
            'stats': self.stats,
        }

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()
        
        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{self.ckpt_path}/{self.name}_ep{self.epoch:04d}.pth"

            self.stats["checkpoints"].append(file_path)

            if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                old_ckpt = self.stats["checkpoints"].pop(0)
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)

            torch.save(state, file_path)

        else:    
            if len(self.stats["results"]) > 0:
                if self.stats["best_result"] is None or self.stats["results"][-1] < self.stats["best_result"]:
                    self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results 
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    if self.ema is not None:
                        self.ema.restore()
                    
                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")
            
    def load_checkpoint(self, checkpoint=None):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/{self.name}_ep*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                epoch_checkpoint = int(checkpoint.split('_ep')[-1].split('.pth')[0])
                self.global_step = epoch_checkpoint * 100
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)
        
        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")            

        if self.ema is not None and 'ema' in checkpoint_dict:
            self.ema.load_state_dict(checkpoint_dict['ema'])

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        
        if self.optimizer and  'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer, use default.")
        
        # strange bug: keyerror 'lr_lambdas'
        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler, use default.")

        if 'scaler' in checkpoint_dict:
            self.scaler.load_state_dict(checkpoint_dict['scaler'])                