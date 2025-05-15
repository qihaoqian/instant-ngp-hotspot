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
    u_cropped = u[
    resolution//20 : resolution*19//20,
    resolution//20 : resolution*19//20,
    resolution//20 : resolution*19//20
    ]
    #print(u.shape, u.max(), u.min(), np.percentile(u, 50))
    
    vertices, triangles = mcubes.marching_cubes(u_cropped , threshold)
    offset = resolution // 20
    vertices = vertices + offset  

    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


def plot_sdf_slice(points, sdfs, plane='z', value=0.0, tolerance=0.02, cmap='coolwarm'):
    """
    Visualize an SDF slice near a given plane (e.g., z=0).

    Parameters:
      points: (N, 3) array of point coordinates.
      sdfs: (N,) or (N, 1) array of SDF values.
      plane: 'x', 'y', or 'z' — specifies the plane for the slice.
      value: The value of the plane, e.g., z=0.
      tolerance: Acceptable offset range, e.g., ±0.02.
      cmap: Colormap to use for visualizing SDF values.
    """
    sdfs = sdfs.flatten()
    axis_idx = {'x': 0, 'y': 1, 'z': 2}[plane]
    
    # Create a mask to filter points near the specified plane
    mask = (
        (np.abs(points[:, axis_idx] - value) < tolerance)  # 靠近平面
        & (points[:, 0] >= -0.9) & (points[:, 0] <= 0.9)    # x 在 [-0.9, 0.9]
        & (points[:, 1] >= -0.9) & (points[:, 1] <= 0.9)    # y 在 [-0.9, 0.9]
    )
    slice_points = points[mask]
    slice_sdfs   = sdfs[mask]
    
    # Determine the axes for the 2D slice visualization
    if plane == 'z':
        x, y = slice_points[:, 0], slice_points[:, 1]
        xlabel, ylabel = 'x', 'y'
    elif plane == 'x':
        x, y = slice_points[:, 1], slice_points[:, 2]
        xlabel, ylabel = 'y', 'z'
    elif plane == 'y':
        x, y = slice_points[:, 0], slice_points[:, 2]
        xlabel, ylabel = 'x', 'z'

    # Plot the 2D slice with SDF values as colors
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(x, y, c=slice_sdfs, cmap=cmap, s=5)
    plt.colorbar(sc, label='SDF Value')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"SDF Slice on {plane}={value:.2f}")
    plt.axis('equal')
    plt.grid(True)
    plt.savefig(f'sdf_slice_{plane}_{value}.png', dpi=300)

class Trainer(object):
    def __init__(self, 
                 name, # name of this experiment
                 model, # network 
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 optimizer=None, # optimizer
                 ema_decay=None, # if use EMA, set the decay
                 lr_scheduler=None, # scheduler
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
                 eikonal_loss_weight=5e1, # weight for eikonal loss
                 sign_loss_weight=1e2, # weight for sign loss
                 heat_loss_weight=5e1, # weight for heat loss
                 h=1e-4, # step size for finite difference
                 heat_loss_lambda=None # lambda for heat loss
                 ):
        
        self.name = name
        self.mute = mute
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
        self.eikonal_loss_weight = eikonal_loss_weight
        self.heat_loss_weight = heat_loss_weight
        self.sign_loss_weight = sign_loss_weight
        self.h = h
        self.heat_loss_lambda = heat_loss_lambda

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

    ### ------------------------------	
    def train_step(self, data):
        # assert batch_size == 1
        X_surf = data["points_surf"][0] # [B, 3]
        y_surf = data["sdfs_surf"][0] # [B]
        X_occ = data["points_occupied"][0] # [B, 3], inside the object
        y_occ = data["sdfs_occupied"][0] # [B]
        X_free = data["points_free"][0] # [B, 3], outside the object
        y_free = data["sdfs_free"][0] # [B]

        X = torch.cat([X_surf, X_occ, X_free], dim=0)
        y = torch.cat([y_surf, y_occ, y_free], dim=0)
        
        # X.requires_grad_(True)
        
        y_pred = self.model(X)
        
        def finite_diff_grad(model, X, h=1e-4):
            # X: [B, 3] Assume input is 3D coordinates
            grads = []
            for i in range(X.shape[1]):
                # Create a zero tensor with the same shape as X
                offset = torch.zeros_like(X)
                offset[:, i] = h  # Only increase the i-th dimension by a small offset
                # Calculate SDF values with positive and negative perturbations
                sdf_plus = model(X + offset)
                sdf_minus = model(X - offset)
                # Calculate the gradient in the i-th direction using central difference formula
                grad_i = (sdf_plus - sdf_minus) / (2 * h)
                grads.append(grad_i.unsqueeze(-1))  # Adjust the dimension for later concatenation
            # Concatenate the gradients in each direction to get a [B, 3] gradient tensor
            grad = torch.cat(grads, dim=-1)
            return grad

        
        def heat_loss(preds, grads=None, sample_pdfs=None, heat_lambda=4):
            heat = torch.exp(-heat_lambda * preds.abs())
            loss = 0.5 * heat**2 * (grads.norm(2, dim=-1) ** 2 + 1)

            if sample_pdfs is not None:
                sample_pdfs = sample_pdfs.squeeze(-1)
                loss = loss.squeeze(-1)
                loss /= sample_pdfs
            loss = loss.sum()

            return loss
        
        
        # mape loss
        surf_pred = y_pred[:X_surf.shape[0]]
        difference = (surf_pred-y_surf).abs()
        scale = 1 / (y_surf.abs() + 1e-2)
        loss_mape = difference * scale
        loss_mape = loss_mape.mean()
            
        # Calculate boundary loss
        if self.boundary_loss_weight != 0:
            loss_boundary = (surf_pred-y_surf).abs().mean()
        else:
            loss_boundary = torch.tensor(0.0, device=y_pred.device)
        
        # Calculate positive_sign_constraint loss
        if self.sign_loss_weight != 0:
            free_pred = y_pred[X_surf.shape[0]+X_occ.shape[0]:]
            beta      = 10.0
            margin    = 0.0
            # free_pred = y_pred[len(X_surf) + len(X_occ):]
            # loss_sign    = F.softplus(-(free_pred - margin) * beta, beta=1.0).mean() / beta
            loss_sign = torch.exp(-1e2 * free_pred).mean()
        else:
            loss_sign = torch.tensor(0.0, device=y_pred.device)
        
        if not (self.eikonal_loss_weight == 0 and self.heat_loss_weight == 0):
            # # Calculate gradient using finite difference method
            gradients = finite_diff_grad(self.model, X, h=self.h)
            
            # # use autograd
            # gradients = torch.autograd.grad(
            #     y_pred, X,
            #     grad_outputs=torch.ones_like(y_pred),
            #     create_graph=True, retain_graph=True, only_inputs=True
            # )[0]
            grad_norm = gradients.norm(2, dim=-1)
        

        # Eikonal loss
        if self.eikonal_loss_weight != 0:
            loss_eikonal = (grad_norm - 1).abs().mean()
        else:
            loss_eikonal = torch.tensor(0.0, device=y_pred.device)
        
        # Heat loss
        if self.heat_loss_weight != 0:
            grad_space = gradients[X_surf.shape[0]:]
            X_space = X[X_surf.shape[0]:]
            space_pred = y_pred[X_surf.shape[0]:]
        
            # lam = self.heat_loss_lambda * self.epoch #decay
            loss_heat = heat_loss(preds=space_pred, grads=grad_space, sample_pdfs=None, heat_lambda=self.heat_loss_lambda)
            loss_heat /= X_space.reshape(-1, 3).shape[0] # average over batch size
        else:
            loss_heat = torch.tensor(0.0, device=y_pred.device)

        # Compute the weighted losses
        weighted_loss_mape = loss_mape * self.mape_loss_weight
        weighted_loss_boundary = loss_boundary * self.boundary_loss_weight
        weighted_loss_eikonal = loss_eikonal * self.eikonal_loss_weight
        weighted_loss_sign = loss_sign * self.sign_loss_weight
        weighted_loss_heat = loss_heat * self.heat_loss_weight

        # Sum up the total loss
        loss = weighted_loss_mape + weighted_loss_boundary + weighted_loss_eikonal + weighted_loss_sign  + weighted_loss_heat

        return y_pred, y, loss, loss_mape, loss_boundary, loss_eikonal, loss_sign, loss_heat

    def eval_step(self, data):
        return self.train_step(data)

    def test_step(self, data):  
        X_surf = data["points_surf"][0] # [B, 3]
        X_free = data["points_free"][0] # [B, 3]

        X = torch.cat([X_surf, X_free], dim=0)
        pred = self.model(X)
        return pred        

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
        
        def get_sdfs_cross_section(self, bound_min, bound_max, resolution, query_func):
            """
            Extracts the cross-section of the xz-plane from the 3D grid data of SDFs and visualizes it for debugging.

            Parameters:
                bound_min: A tuple (x_min, y_min, z_min) representing the minimum coordinates of the grid region.
                bound_max: A tuple (x_max, y_max, z_max) representing the maximum coordinates of the grid region.
                resolution: A tuple (nx, ny, nz) representing the resolution in each direction.
                query_func: A function used to compute the SDF value for each point (passed to extract_fields).

            Returns:
                cross_section: A 2D numpy array of the xz-plane, with shape (nx, nz).
            """
            # Extract the SDF values for the entire grid, assuming extract_fields is implemented
            u = extract_fields(bound_min, bound_max, resolution, query_func)
            
            # Assuming u has the shape (nx, ny, nz), select the middle y layer as the cross-section
            nz = u.shape[2]
            z_index = nz // 2  # Select the middle layer
            cross_section = u[resolution//8:resolution*7//8, resolution//8:resolution*7//8, z_index]  # Resulting shape is (nx, nz)
            
            # Compute the physical coordinate range in the x and z directions for the extent parameter in the image
            x_min, y_min, z_min = bound_min * 0.9
            x_max, y_max, z_max = bound_max * 0.9
            extent = [x_min, x_max, y_min, y_max]
            
            # Visualize the cross-section using a color map to distinguish different SDF values
            plt.figure(figsize=(8, 6))
            # Transpose the cross-section to match x-axis as horizontal, z-axis as vertical, and set origin='lower'
            plt.imshow(cross_section.T, origin='lower', extent=extent, cmap='jet')
            plt.colorbar(label='SDF Distance')
            plt.xlabel('x')
            plt.ylabel('y')

            plt.title(f'SDFs xy Cross Section (z = {z_index:.2f} at epoch{self.epoch})')
            # Save the figure to a buffer and add it to TensorBoard
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            image = Image.open(buf)
            image = np.array(image)
            # Assuming 'writer' is your tensorboard SummaryWriter and is available in the scope.
            self.writer.add_image(f'sdfs_cross_section_xy/epoch{self.epoch}', image, self.epoch, dataformats='HWC')
            buf.close()
            # plt.show()

        get_sdfs_cross_section(self, bounds_min, bounds_max, resolution, query_func)
        vertices, triangles = extract_geometry(bounds_min, bounds_max, resolution=resolution, threshold=0, query_func=query_func)
        print(f"==> vertices: {vertices.shape}, triangles: {triangles.shape}")

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
                preds, truths, loss, loss_mape, loss_boundary, loss_eikonal, loss_sign, loss_heat = self.train_step(data)
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
                    self.writer.add_scalar("train/loss_eikonal", loss_eikonal.item(), self.global_step)
                    self.writer.add_scalar("train/loss_sign", loss_sign.item(), self.global_step)
                    self.writer.add_scalar("train/loss_heat", loss_heat.item(), self.global_step)

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