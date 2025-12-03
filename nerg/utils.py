from matplotlib import pyplot as plt

from common.logging import log

import numpy as np

import os
import tqdm
import tensorboardX

from common.timer import Timer
from common.viridis_cm import apply_viridis_cm
from data_gaze.dataset import GazeDataset
from data_gaze.gaze_probe import GazeProbePlotHelper, plot_gaze_rays_p, plot_gaze_rays
from data_gaze.data_skel_tracking.head_pose_loader import plot_gaze_probe_centers, pos_nerf2gaze, dir_nerf2gaze
from nerf_instant_ngp.nerf.utils import Trainer, get_rays

import torch
import torch.nn.functional as f

from nerg.gaze_prediction import NeRGRenderer


def loss_pears(x, y, eps=1e-6):
    xm = x - x.mean()
    ym = y - y.mean()
    normxm = torch.linalg.norm(xm) + eps
    normym = torch.linalg.norm(ym) + eps
    r = torch.dot(xm / normxm, ym / normym)
    r = torch.clamp(r, 0, 1)
    return 1 - r


def loss_mae(y_pred, y_gt):
    return torch.mean(torch.abs(y_gt - y_pred))


def loss_sphere_cdf(y_pred):
    # CDF over the sphere must be 1.0
    # integration with unbiased Monte-Carlo estimator (assuming uniform samples)
    cdf = 4 * torch.pi * torch.mean(y_pred)
    return (1 - cdf) ** 2


def loss_kldiv(y_pred, y_gt, eps=1e-6):
    return torch.mean(y_gt * torch.log(eps + y_gt / (eps + y_pred)))


def compute_nerg_losses(y_pred, y_gt):
    l_mae = loss_mae(y_pred.view(-1), y_gt.view(-1))
    l_pearson = loss_pears(y_pred.view(-1), y_gt.view(-1))
    l_kldiv = loss_kldiv(y_pred.view(-1), y_gt.view(-1))
    l_cdf = loss_sphere_cdf(y_pred.view(-1))

    # loss = l_mae + l_pearson + l_kldiv  # seems less stable
    loss = l_mae + 0.1 * l_pearson

    return {
        "loss": loss,
        "l_mae": l_mae,
        "l_pearson": l_pearson,
        "l_kldiv": l_kldiv,
        "l_cdf": l_cdf,
    }


class TrainerNeRG(Trainer):
    def __init__(self, name, opt, model: NeRGRenderer, **kwargs):
        super().__init__(name, opt, model, **kwargs)
        self.criterion = compute_nerg_losses
        self.error_map = None
        self.model_nerf = model.model_nerf

        if self.workspace is not None:
            os.makedirs(f'{self.workspace}/plots', exist_ok=True)

    def train_gui(self, train_loader, step):
        # TODO not required for now...
        raise NotImplementedError()

    def train(self, train_dataset: GazeDataset, train_loader, valid_loader, max_epochs):
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        print(f"Start training with max_epochs={max_epochs}")
        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch

            # this will repeatedly call nerg_trainer.train_step() defined below
            self.train_one_epoch(train_loader)

            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint(full=False, best=True)

            if self.model.use_gaze_smoothing:
                print('model_nerg.gaze_smoothing_sigma', self.model.gaze_smoothing_sigma)

            # plot the GazeProbes to visualize the training data
            plot_gaze_probe_centers(train_dataset.gaze_probes, save_path=f'{self.workspace}/plots', tag=f"Epoch {epoch} coords", nerf_coords=False)

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def train_step(self, data):
        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]
        gaze_prob_gt = data['gaze_prob']  # [B, N]

        outputs = self.model.predict_gaze(
            rays_o, rays_d, perturb=True, max_steps=self.opt.max_steps, dt_gamma=self.opt.dt_gamma, gaze_smoothing=True)

        gaze_prob = outputs["gaze_prob"]

        loss = compute_nerg_losses(gaze_prob, gaze_prob_gt)

        return gaze_prob, gaze_prob_gt, loss

    def eval_step(self, data):
        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]
        gaze_prob_gt = data['gaze_prob']  # [B, N]

        outputs = self.model.predict_gaze(
            rays_o, rays_d, perturb=False, max_steps=self.opt.max_steps, dt_gamma=self.opt.dt_gamma, gaze_smoothing=True)

        gaze_prob = outputs["gaze_prob"]
        pred_depth = outputs['depth']

        loss = compute_nerg_losses(gaze_prob, gaze_prob_gt)

        return gaze_prob, pred_depth, gaze_prob_gt, loss

    def test_step(self, data, bg_color=None, perturb=False, gaze_smoothing=False):
        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]
        H, W = data['H'], data['W']

        outputs = self.model.predict_gaze(
            rays_o, rays_d, perturb=perturb, max_steps=self.opt.max_steps, dt_gamma=self.opt.dt_gamma,
            gaze_smoothing=gaze_smoothing,
        )

        gaze_prob = outputs["gaze_prob"]  # first channel from rgb output

        gaze_prob = gaze_prob.reshape(-1, H, W)
        pred_depth = outputs['depth'].reshape(-1, H, W)
        pred_depth_w = outputs['depth_w'].reshape(-1, H, W)

        return gaze_prob, pred_depth, pred_depth_w

    # moved out bg_color and perturb for more flexible control...
    def render_nerf(self, data, bg_color=None, perturb=False, return_geofeat=False):
        return super().test_step(data, bg_color, perturb, model=self.model_nerf, return_geofeat=return_geofeat)

    # [GUI] test on a single image
    def test_gui(self, pose, intrinsics, W, H, bg_color=None, spp=1, downscale=1):
        # render resolution (may need downscale to for better frame rate)
        rH = int(H * downscale)
        rW = int(W * downscale)
        intrinsics = intrinsics * downscale

        pose = torch.from_numpy(pose).unsqueeze(0).to(self.device)
        rays = get_rays(pose, intrinsics, rH, rW, -1)
        data = {
            'rays_o': rays['rays_o'].clone(),
            'rays_d': rays['rays_d'],
            'H': rH,
            'W': rW,
        }

        self.model.eval()
        self.model_nerf.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.fp16):
                # here spp is used as perturb random seed! (but not perturb the first sample)
                pred_gaze, pred_depth, pred_depth_w = self.test_step(
                    data, perturb=False if spp == 1 else spp,
                    gaze_smoothing=False  # disable gaze smoothing because the resolution is too large
                )

        if self.ema is not None:
            self.ema.restore()

        # interpolation to the original resolution
        if downscale != 1:
            # torchngp TODO: have to permute twice with torch...
            pred_gaze = f.interpolate(pred_gaze.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)
            pred_depth = f.interpolate(pred_depth.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)
            pred_depth_w = f.interpolate(pred_depth_w.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)

        pred_gaze = pred_gaze[0].detach().cpu().numpy()
        pred_depth = pred_depth[0].detach().cpu().numpy()
        pred_depth_w = pred_depth_w[0].detach()

        pred_gaze -= pred_gaze.min()
        pred_gaze /= pred_gaze.max() + 1e-6

        if self.opt.gaze_cmap == "viridis":
            pred_rgb = apply_viridis_cm(np.nan_to_num(pred_gaze, posinf=1.0))
        else:
            pred_rgb = np.zeros((*pred_gaze.shape, 3), float)
            pred_rgb[..., :] = pred_gaze

        # follow torch-ngp output dict format
        outputs = {
            # tensors
            'data': data,
            'depth_w': pred_depth_w,
            # tensors
            'image': pred_rgb,
            'depth': pred_depth,
        }

        return outputs

    def load_checkpoint(self, checkpoint=None, model_only=False):
        def replace_keys(checkpoint_dict):
            print('WARNING: load_checkpoint replacing keys...')
            def replace_key(d, k1, k2):
                if k1 in d:
                    d[k2] = d.pop(k1)
            replace_key(checkpoint_dict['model'], 'lin_emit.weight', 'emit_head.weight')
            replace_key(checkpoint_dict['model'], 'lin_capture.weight', 'capture_head.weight')

        checkpoint_dict = super().load_checkpoint(checkpoint, model_only, checkpoint_func=replace_keys)
        if checkpoint_dict is not None:
            if 'variant' in checkpoint_dict:
                if checkpoint_dict['variant'] != self.model.variant:
                    raise ValueError("Checkpoint model variant does not match model variant")
        else:
            print("WARNING: Checkpoint does not have NeRG.variant.")

    def save_checkpoint(self, name=None, full=False, best=False, remove_old=True):
        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
            'variant': self.model.variant,
        }

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()

        # save ema results
        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        state['model'] = self.model.state_dict()

        if self.ema is not None:
            self.ema.restore()
        file_path = f"{self.ckpt_path}/{name}.pth"

        if remove_old:
            self.stats["checkpoints"].append(file_path)

            if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                old_ckpt = self.stats["checkpoints"].pop(0)
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)

        torch.save(state, file_path)


def evaluate_nerg_on_dataset_images(trainer: TrainerNeRG, loader, save_path=None, name=None, num_images_eval=8, tag=""):
    if save_path is None:
        save_path = os.path.join(trainer.workspace, 'results')
        log(f"evaluate_nerg_on_dataset_images save_path={save_path}")

    if name is None:
        name = f'{trainer.name}_ep{trainer.epoch:04d}'

    os.makedirs(save_path, exist_ok=True)

    trainer.log(f"==> Start Test, save results to {save_path}")

    pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                     bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    trainer.model.eval()

    for i, data in enumerate(loader):
        if num_images_eval < i:
            break
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=trainer.fp16):
                with Timer(name="evaluate_nerg_on_dataset_images"):
                    preds, _, _ = trainer.test_step(data, gaze_smoothing=True)
                preds_nerf, _, _ = trainer.render_nerf(data)

            rays_o = data["rays_o"].detach().cpu().numpy()  # evaluated gaze ray position
            rays_o = rays_o.reshape(-1, 3).mean(axis=0).reshape(3, 1)  # reshape(3,1) = {1,3}.transpose()
            rays_o = pos_nerf2gaze(rays_o).reshape(3)
            ro_str = [f'{rays_o[i]:.3f}' for i in range(3)]  # query center position as str

            rays_d = data["rays_d"].detach().cpu().numpy()  # evaluated gaze rays
            rays_d = dir_nerf2gaze(rays_d.reshape(-1, 3).transpose())  # 3,N

            # raise NotImplementedError()

            gaze_prob_gt = data["gaze_prob"].detach().cpu().numpy().flatten()
            gaze_prob_pred = preds.detach().cpu().numpy().flatten()

        # save figures
        with GazeProbePlotHelper(os.path.join(save_path, f'{tag}{name}_{i:04d}_gaze_pred.png'), legend=True):
            plt.title(f"Predicted gaze kernel density ro={ro_str}")
            # plot_gaze_rays(rays_d, color='r', label='rays')
            plot_gaze_rays_p(rays_d, gaze_prob_pred)
            print('plotted gaze_prob_pred', gaze_prob_pred.shape, gaze_prob_pred.min(), gaze_prob_pred.mean(),
                  gaze_prob_pred.max())
            plot_gaze_rays(rays_d, color='r', label='centers')

        with GazeProbePlotHelper(os.path.join(save_path, f'{tag}{name}_{i:04d}_gaze_gt.png'), legend=True):
            plt.title(f"Ground truth gaze kernel density ro={ro_str}")
            # plot_gaze_rays(rays_d, color='r', label='rays')
            plot_gaze_rays_p(rays_d, gaze_prob_gt)
            print('plotted gaze_prob_gt', gaze_prob_gt.shape, gaze_prob_gt.min(), gaze_prob_gt.mean(),
                  gaze_prob_gt.max())
            plot_gaze_rays(rays_d, color='r', label='centers')

        pbar.update(loader.batch_size)
        plt.show()

    log("Finished evaluate_nerg_on_dataset_images()")
