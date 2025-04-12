import torch
import argparse

from hotspot.utils import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-3, help="initial learning rate")

    opt = parser.parse_args()
    print(opt)

    seed_everything(opt.seed)

    from hotspot.netowrk import SDFNetwork

    model = SDFNetwork(encoding="hashgrid")
    print(model)

    if opt.test:
        trainer = Trainer('ngp', model, workspace=opt.workspace, use_checkpoint='best', eval_interval=1, boundary_loss_weight=1e7, eikonal_loss_weight=1e-4, h=1e-3)
        trainer.save_mesh(os.path.join(opt.workspace, 'results', 'output.ply'), 1024)

    else:
        from hotspot.dataset import SDFDataset

        train_dataset = SDFDataset(opt.path, size=100, num_samples=2**18)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

        valid_dataset = SDFDataset(opt.path, size=1, num_samples=2**18) # just a dummy
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1)

        optimizer = lambda model: torch.optim.Adam([
            {'name': 'encoding', 'params': model.encoder.parameters()},
            {'name': 'net', 'params': model.backbone.parameters(), 'weight_decay': 1e-6},
        ], lr=opt.lr, betas=(0.9, 0.99), eps=1e-15)

        scheduler = lambda optimizer: optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        trainer = Trainer('ngp', model, workspace=opt.workspace, optimizer=optimizer, ema_decay=0.95, 
                          lr_scheduler=scheduler, use_checkpoint='latest', eval_interval=5, 
                          boundary_loss_weight=3e3, eikonal_loss_weight=5e1, sign_loss_weight=1e2, heat_loss_weight=5e-1, h=1e-4)

        trainer.train(train_loader, valid_loader, 20)

        # also test
        trainer.save_mesh(os.path.join(opt.workspace, 'results', 'output.ply'), resolution=512)
