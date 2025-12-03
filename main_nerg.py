"""
notes:
    nerf model add if x0<x<x1 bound, output red color, test gaze vs nerf positions
    maybe transformation matrices are not correct
"""

from transformers import get_scheduler

import argparse
import pickle as pkl

from data_gaze.dataset import GazeDataset, GAZE_QUERY_TYPE_RADIUS
from common.lock_remover import check_lock_files
from main_nerf import get_nerf_network
from nerf_instant_ngp.nerf.renderer import NeRFRenderer
from nerg.gui import NeRFGUI
from nerf_instant_ngp.nerf.utils import *

from common.logging import log
from nerg.utils import evaluate_nerg_on_dataset_images, TrainerNeRG
from nerg.gaze_prediction import NeRGRenderer

#torch.autograd.set_detect_anomaly(True)


__DATASET_TRAIN_PKL = "train_dataset.pkl"
__DATASET_VAL_PKL = "val_dataset.pkl"


def get_datasets(workspace, device, load_pkl=True, save_pkl=True, return_gaze_probe=False, preload_gaze_rays=True):
    log("Loading gaze data...")

    def load_dataset_pkl(filename):
        try:
            with open(f"{workspace}/{filename}", "rb") as dataset_file:
                dataset = pkl.load(dataset_file)
            log(f"Loaded precomputed gaze dataset {filename}.")
            return dataset
        except FileNotFoundError:
            return None

    train_dataset = load_dataset_pkl(__DATASET_TRAIN_PKL) if load_pkl else None
    if train_dataset is None:
        log("Recomputing train dataset...")
        train_dataset = GazeDataset(
            device=device,
            gaze_query_type=GAZE_QUERY_TYPE_RADIUS,
            gaze_probe_count=opt.gaze_count,
            # gaze_spheres_count=1,  # debug: overfit on one gaze sphere!
            gaze_query_radius=opt.gaze_radius,
            preload_gaze_probes=opt.gaze_precompute,
            # gaze_debug_load_lines=50000,
        )
        if save_pkl:
            with open(f"{workspace}/{__DATASET_TRAIN_PKL}", "wb") as dataset_file:
                pkl.dump(train_dataset, dataset_file)

    valid_dataset = load_dataset_pkl(__DATASET_VAL_PKL) if load_pkl else None
    if valid_dataset is None:
        log("Recomputing validation dataset...")
        valid_dataset = GazeDataset(
            device=device,
            gaze_query_type=GAZE_QUERY_TYPE_RADIUS,
            gaze_query_radius=opt.gaze_radius,
            gaze_probe_count=opt.gaze_count_eval,
            preload_gaze_probes=opt.gaze_precompute,
        )
        if save_pkl:
            with open(f"{workspace}/{__DATASET_VAL_PKL}", "wb") as dataset_file:
                pkl.dump(valid_dataset, dataset_file)

    log("Train dataset preload_gaze_probes =", train_dataset.preload_gaze_probes)

    # collate function disables
    train_dataset.return_gaze_probe = return_gaze_probe
    train_dataset.preload_gaze_rays = preload_gaze_rays

    valid_dataset.return_gaze_probe = return_gaze_probe

    return train_dataset, valid_dataset


def get_nerg_network(opt, model_nerf: NeRFRenderer, remove_locks=True):
    if remove_locks:
        # possibly remove lock files to avoid a deadlock during model creation
        check_lock_files(remove=True)

    return NeRGRenderer(
        model_nerf,
        bound=opt.bound,
        min_near=opt.min_near,
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --preload")
    parser.add_argument('--nerg', action='store_true', help="render NeRF and NeRG together")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)

    ### training options
    parser.add_argument('--iters', type=int, default=500000, help="training iters")
    parser.add_argument('--lr', type=float, default=1e-3, help="initial learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--num_rays', type=int, default=4096, help="num rays sampled per image for each training step")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=1024, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=512, help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0, help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=4096, help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")
    parser.add_argument('--patch_size', type=int, default=1, help="[experimental] render patches in training, so as to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable")

    ### network backbone options
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")
    parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")

    ### NeRG Gaze/attention options
    parser.add_argument('--nerf_path', type=str, default="./results/nerf_long-b5-bg10", help="Path to pre-trained NERF run with checkpoints.")
    parser.add_argument('--gaze_precompute', action='store_true', help="Toggle to preload gaze probes or sample during training")
    parser.add_argument('--gaze_radius', type=float, default=0.1, help="Radius in meters for gaze ray queries/clustering")
    parser.add_argument('--gaze_count', type=int, default=4096, help="Number of gaze probes per epoch (train)")
    parser.add_argument('--gaze_count_eval', type=int, default=512, help="Number of gaze probes per epoch (eval)")
    parser.add_argument('--gaze_cmap', type=str, default='viridis', help="Color map name for NerG output (only when nerf gui is used). Convert from probability density to colors.")
    parser.add_argument('--test_images_path', type=str, default='')

    ### dataset options
    parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
    parser.add_argument('--preload', action='store_true', help="preload all data into GPU, accelerate training but use more GPU memory")
    # (the default value is for the fox dataset)
    parser.add_argument('--bound', type=float, default=2, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=0.33, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--dt_gamma', type=float, default=1/128, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.2, help="minimum near distance for camera")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
    parser.add_argument('--bg_radius', type=float, default=-1, help="if positive, use a background model at sphere(bg_radius)")

    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=1024, help="GUI width")
    parser.add_argument('--H', type=int, default=720, help="GUI height")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=64, help="GUI rendering max sample per pixel")

    ### experimental
    parser.add_argument('--error_map', action='store_true', help="use error map to sample rays")
    parser.add_argument('--clip_text', type=str, default='', help="text input for CLIP guidance")
    parser.add_argument('--rand_pose', type=int, default=-1, help="<0 uses no rand pose, =0 only uses rand pose, >0 sample one rand pose every $ known poses")

    opt = parser.parse_args()

    if opt.O:
        log("Using opt.O")

        opt.fp16 = True
        opt.cuda_ray = True
        opt.preload = True

        opt.gaze_precompute = True

    if opt.gui and opt.test:
        opt.gaze_cmap = "viridis"

    log('opt', opt)

    seed_everything(opt.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set up NeRF model
    # note: since torch-ngp loads NeRF through the Trainer instance which is difficult to set up here,
    # we need load the NeRF model manually instead
    model_nerf = get_nerf_network(opt, device=device, nerf_ckpt=opt.nerf_path)

    # set up NeRG model
    model_nerg = get_nerg_network(opt, model_nerf)
    log('model', model_nerg)
    log("NeRG Total trainable parameters", sum([p.numel() if p.requires_grad else 0 for p in model_nerg.parameters()]))

    if opt.test:
        log("Test only")

        metrics = []

        trainer_nerg = TrainerNeRG(
            'nerg', opt, model_nerg, device=device, workspace=opt.workspace, fp16=opt.fp16, use_checkpoint=opt.ckpt,
        )

        if opt.gui:
            gui = NeRFGUI(opt, trainer_nerg, use_gaze=opt.nerg)
            gui.render()

        else:
            train_dataset, valid_dataset = get_datasets(trainer_nerg.workspace, device)

            evaluate_nerg_on_dataset_images(trainer_nerg, train_dataset.dataloader(), tag="train")
            evaluate_nerg_on_dataset_images(trainer_nerg, valid_dataset.dataloader(), tag="val")

    else:

        log("DO TRAINING")

        optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)

        num_warmup_iters = 128
        scheduler = lambda optimizer: get_scheduler(
            "cosine", optimizer,
            num_warmup_steps=num_warmup_iters,
            num_training_steps=opt.iters + num_warmup_iters  # total steps
        )

        # # decay to 0.1 * init_lr at last iter step
        # scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

        trainer_nerg = TrainerNeRG(
            'nerg', opt, model_nerg, device=device, workspace=opt.workspace, optimizer=optimizer,
            fp16=opt.fp16, lr_scheduler=scheduler,
            scheduler_update_every_step=True, use_checkpoint=opt.ckpt, eval_interval=1,
        )

        train_dataset, valid_dataset = get_datasets(trainer_nerg.workspace, device)
        train_loader = train_dataset.dataloader()
        valid_loader = valid_dataset.dataloader()

        max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
        log(f"Starting training with max_epoch={max_epoch}...")

        if opt.gui:
            raise NotImplementedError("Training in GUI is not implemented for NeRG.")
            # gui = NeRGGUI(opt, trainer_nerg, train_loader)
            # gui.render()

        else:
            trainer_nerg.train(train_dataset, train_loader, valid_loader, max_epoch)
