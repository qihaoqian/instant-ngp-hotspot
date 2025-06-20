import torch
from config.config import Config

from hotspot.utils import *

if __name__ == '__main__':
    parser = Config.get_argparser()
    cfg: Config = parser.parse_args()
    os.makedirs(cfg.trainer.workspace, exist_ok=True)
    cfg.as_yaml(f"{cfg.trainer.workspace}/config.yaml")

    seed_everything(cfg.seed)

    from hotspot.network import SDFNetwork
    
    if cfg.model.encoding == "hashgrid":
        encoding_config = cfg.hash_grid
    elif cfg.model.encoding == "reg_grid":
        encoding_config = cfg.reg_grid
    else:
        encoding_config = None

    model = SDFNetwork(encoding=cfg.model.encoding, 
                    encoding_config=encoding_config,
                    num_layers=cfg.model.num_layers, 
                    hidden_dim=cfg.model.hidden_dim,
                    sphere_radius=cfg.model.sphere_radius,
                    sphere_scale=cfg.model.sphere_scale,
                    use_sphere_post_processing=cfg.model.use_sphere_post_processing,
                    )
    print(model)

    
    if cfg.test:
        trainer = Trainer(
            name=cfg.trainer.name,
            model=model,
            workspace=cfg.trainer.workspace,
            use_checkpoint=cfg.trainer.use_checkpoint,
        )

        trainer.save_mesh(os.path.join(cfg.trainer.workspace, 'results', 'output.ply'), 1024)

    else:
        from hotspot.dataset import SDFDataset

        train_dataset = SDFDataset(cfg.data.dataset_path, size=cfg.data.train_size, num_samples_surf=cfg.data.num_samples_surf,
                                   num_samples_space=cfg.data.num_samples_space)
        train_dataset.plot_all_sdf_combined(workspace=cfg.trainer.workspace, coord=0.0)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

        valid_dataset = SDFDataset(cfg.data.dataset_path, size=cfg.data.valid_size, num_samples_surf=cfg.data.num_samples_surf,
                                   num_samples_space=cfg.data.num_samples_space) # just a dummy
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1)

        optimizer = lambda model: torch.optim.Adam([
            {'name': 'encoding', 'params': model.encoder.parameters()},
            # {'name': 'net', 'params': model.backbone.parameters(), 'weight_decay': cfg.optimizer.weight_decay},
            {'name': 'net', 'params': model.backbone.parameters()},
        ], lr=cfg.optimizer.lr, betas=cfg.optimizer.betas, eps=cfg.optimizer.eps)

        scheduler = lambda optimizer: optim.lr_scheduler.StepLR(optimizer, step_size=cfg.scheduler.step_size, gamma=cfg.scheduler.gamma)

        trainer = Trainer(
            name=cfg.trainer.name,
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            max_epochs=cfg.epochs,
            workspace=cfg.trainer.workspace,
            max_keep_ckpt=cfg.trainer.max_keep_ckpt,
            use_checkpoint=cfg.trainer.use_checkpoint,
            eval_interval=cfg.trainer.eval_interval,
            ema_decay=cfg.trainer.ema_decay,
            use_tensorboardX=cfg.trainer.use_tensorboardX,
            boundary_loss_weight=cfg.trainer.boundary_loss_weight,
            eikonal_loss_surf_weight=cfg.trainer.eikonal_loss_surf_weight,
            eikonal_loss_space_weight=cfg.trainer.eikonal_loss_space_weight,
            sign_loss_free_weight=cfg.trainer.sign_loss_free_weight,
            sign_loss_occ_weight=cfg.trainer.sign_loss_occ_weight,
            heat_loss_weight=cfg.trainer.heat_loss_weight,
            projection_loss_weight=cfg.trainer.projection_loss_weight,
            grad_dir_loss_weight=cfg.trainer.grad_direction_loss_weight,
            sec_grad_loss_weight=cfg.trainer.second_gradient_loss_weight,
            small_sdf_loss_weight=cfg.trainer.small_sdf_loss_weight,
            h1=cfg.trainer.h1,
            h2=cfg.trainer.h2,
            heat_loss_lambda=cfg.trainer.heat_loss_lambda,
        )

        trainer.train(train_loader, valid_loader, cfg.epochs)

        # also test
        trainer.save_mesh(os.path.join(cfg.trainer.workspace, 'results', 'output.ply'), resolution=cfg.trainer.resolution)

        # Loss visualization after training
        print("开始执行损失可视化...")

        from hotspot.loss_visualizer import visualize_all_losses_slices
        
        # 构造checkpoint路径
        checkpoint_path = os.path.join(cfg.trainer.workspace, 'checkpoints', f'ngp_ep{cfg.epochs:04d}.pth')
        
        # 构造配置文件路径
        config_path = os.path.join(cfg.trainer.workspace, 'config.yaml')
        
        # 损失可视化保存目录
        loss_vis_dir = os.path.join(cfg.trainer.workspace, 'loss_visualizations')
        
        # 执行损失可视化
        visualize_all_losses_slices(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            mesh_path=cfg.data.dataset_path,
            save_dir=loss_vis_dir,
            resolution=128,
            epoch=cfg.epochs
        )
        print(f"损失可视化完成，结果保存在: {loss_vis_dir}")
            

