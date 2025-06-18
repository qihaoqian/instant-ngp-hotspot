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
        train_dataset.plot_all_sdf_slices(workspace=cfg.trainer.workspace) # plot ground truth SDF slice
        train_dataset.plot_all_sdf_binary_slices(workspace=cfg.trainer.workspace) # plot ground truth SDF binary slice
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

        valid_dataset = SDFDataset(cfg.data.dataset_path, size=cfg.data.valid_size, num_samples_surf=cfg.data.num_samples_surf,
                                   num_samples_space=cfg.data.num_samples_space) # just a dummy
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1)

        optimizer = lambda model: torch.optim.Adam([
            {'name': 'encoding', 'params': model.encoder.parameters()},
            {'name': 'net', 'params': model.backbone.parameters(), 'weight_decay': cfg.optimizer.weight_decay},
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
            h1=cfg.trainer.h1,
            h2=cfg.trainer.h2,
            heat_loss_lambda=cfg.trainer.heat_loss_lambda,
        )

        trainer.train(train_loader, valid_loader, cfg.epochs)

        # also test
        trainer.save_mesh(os.path.join(cfg.trainer.workspace, 'results', 'output.ply'), resolution=cfg.trainer.resolution)
