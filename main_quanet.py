import torch
from config.config import Config
from config.config_abc import ConfigABC

from hotspot.utils import *

if __name__ == '__main__':
    parser = Config.get_argparser()
    cfg: Config = parser.parse_args()
    
    seed_everything(cfg.seed)

    # from hotspot.netowrk import SDFNetwork
    from hotspot.QuaNet import Network
     
    model = Network(
        latent_size=0,
        in_dim=3,
        decoder_hidden_dim=128,
        nl='sine',
        encoder_type='none',
        decoder_n_hidden_layers=5,
        init_type='mfgi',
        neuron_type='quadratic',
        sphere_init_params=[1.6, 0.1],
        n_repeat_period=30,
    ) 
    
    print(model)

    if cfg.test:
        trainer = Trainer(
            name=cfg.trainer.name,
            model=model,
            workspace=cfg.trainer.workspace,
            use_checkpoint=cfg.trainer.use_checkpoint,
            eval_interval=cfg.trainer.eval_interval,
            ema_decay=cfg.trainer.ema_decay,
            use_tensorboardX=cfg.trainer.use_tensorboardX,
            boundary_loss_weight=cfg.trainer.boundary_loss_weight,
            eikonal_loss_weight=cfg.trainer.eikonal_loss_weight,
            sign_loss_weight=cfg.trainer.sign_loss_weight,
            heat_loss_weight=cfg.trainer.heat_loss_weight,
            h=cfg.trainer.h,
            heat_loss_lambda=cfg.trainer.heat_loss_lambda,
        )
                    
        trainer.save_mesh(os.path.join(cfg.trainer.workspace, 'results', 'output.ply'), 1024)

    else:
        from hotspot.dataset import SDFDataset

        train_dataset = SDFDataset(cfg.data.dataset_path, size=cfg.data.train_size, num_samples=cfg.data.num_samples)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

        valid_dataset = SDFDataset(cfg.data.dataset_path, size=cfg.data.valid_size, num_samples=cfg.data.num_samples) # just a dummy
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1)

        optimizer = lambda model: torch.optim.Adam([
            {'name': 'decoder', 'params': model.decoder.parameters()},
        ], lr=cfg.optimizer.lr, betas=cfg.optimizer.betas, eps=cfg.optimizer.eps)

        scheduler = lambda optimizer: optim.lr_scheduler.StepLR(optimizer, step_size=cfg.scheduler.step_size, gamma=cfg.scheduler.gamma)

        trainer = Trainer(
            name=cfg.trainer.name,
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            workspace=cfg.trainer.workspace,
            use_checkpoint=cfg.trainer.use_checkpoint,
            eval_interval=cfg.trainer.eval_interval,
            ema_decay=cfg.trainer.ema_decay,
            use_tensorboardX=cfg.trainer.use_tensorboardX,
            boundary_loss_weight=cfg.trainer.boundary_loss_weight,
            eikonal_loss_weight=cfg.trainer.eikonal_loss_weight,
            sign_loss_weight=cfg.trainer.sign_loss_weight,
            heat_loss_weight=cfg.trainer.heat_loss_weight,
            h=cfg.trainer.h,
            heat_loss_lambda=cfg.trainer.heat_loss_lambda,
        )

        trainer.train(train_loader, valid_loader, cfg.epochs)

        # also test
        trainer.save_mesh(os.path.join(cfg.trainer.workspace, 'results', 'output.ply'), resolution=512)
