"""
Inspired by Instant-NGP implementation for NeRFs from https://github.com/ashawkey/torch-ngp
"""
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from enum import Enum, unique

from timm.models.vision_transformer import Block as ViTBlock
from common.logging import log_warn
from common.vector_math import magnitude_t, lerp
from data_gaze.dataset import GazeDataset, GAZE_QUERY_TYPE_COUNT
from data_gaze.gaze_probe import GazeProbe
from data_gaze.data_skel_tracking.head_pose_loader import pos_nerf2gaze, dir_nerf2gaze
from data_gaze.utils import unit_sphere_smoothing
from nerf_instant_ngp.nerf.renderer import NeRFRenderer
from nerf_instant_ngp.encoding import get_encoder

from nerg.deepgaze_wrapper import DeepGazeWrapper
from nerg.vmf_mixture import VMFHead, vmf_log_density_mixture

DROPOUT_HIDDEN = 0.15
DROPOUT_GEOFEAT = 0.
DROPOUT_DIRFEAT = 0.


def make_nets_nerf(encoder_dim_out, hidden_dim_net1, geo_feat_dim, encoder_dir_dim_out, hidden_dim_net2):
    # simple MLP-based networks similar to the original implementation of torch-ngp (see nerf/network.py)
    net1 = nn.Sequential(
        nn.Linear(encoder_dim_out, hidden_dim_net1, bias=False),
        nn.ReLU(),
        nn.Dropout(DROPOUT_HIDDEN),
        nn.Linear(hidden_dim_net1, geo_feat_dim, bias=False)
    )

    in_dim2 = encoder_dir_dim_out + geo_feat_dim
    net2 = nn.Sequential(
        nn.Linear(in_dim2, hidden_dim_net2, bias=False),
        nn.ReLU(),
        nn.Dropout(DROPOUT_HIDDEN),
        nn.Linear(hidden_dim_net2, hidden_dim_net2, bias=False)
    )

    return net1, net2


def make_nets_deepgaze2e(encoder_dim_out, hidden_dim_net1, geo_feat_dim, encoder_dir_dim_out, hidden_dim_net2):
    # inspired by DeepGaze2e readout network modified for NeRF-style inputs
    net1 = nn.Sequential(OrderedDict([
        ('layernorm0', nn.LayerNorm(encoder_dim_out)),
        ('lin0', nn.Linear(encoder_dim_out, hidden_dim_net1, bias=True)),
        ('act0', nn.ReLU()),
        ('drop0', nn.Dropout(DROPOUT_HIDDEN)),

        ('layernorm1', nn.LayerNorm(hidden_dim_net1)),
        ('lin1', nn.Linear(hidden_dim_net1, geo_feat_dim, bias=True)),
        ('act1', nn.ReLU()),
        ('drop1', nn.Dropout(DROPOUT_GEOFEAT)),
    ]))  # 2 layers

    in_dim2 = encoder_dir_dim_out + geo_feat_dim
    net2 = nn.Sequential(OrderedDict([
        ('layernorm0', nn.LayerNorm(in_dim2)),
        ('lin0', nn.Linear(in_dim2, hidden_dim_net2, bias=True)),
        ('act0', nn.ReLU()),
        ('drop0', nn.Dropout(DROPOUT_HIDDEN)),

        ('layernorm1', nn.LayerNorm(hidden_dim_net2)),
        ('lin1', nn.Linear(hidden_dim_net2, hidden_dim_net2, bias=True)),
        ('act1', nn.ReLU()),
        ('drop1', nn.Dropout(DROPOUT_DIRFEAT)),
    ]))  # 2 layers

    return net1, net2


class AttendBlock(nn.Module):
    def __init__(self, hidden_dim=64, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.block = ViTBlock(hidden_dim, num_heads=num_heads, mlp_ratio=1)
        self.block.attn.fused_attn = False
        self.head = nn.Linear(hidden_dim, 1)
        self.gaze_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))

    def forward(self, h_emit, h_capture):
        num_rays = h_emit.shape[0]  # shape is B,D
        gaze_tokens = self.gaze_token.expand(num_rays, 1, self.hidden_dim)

        h = torch.cat([
            gaze_tokens,
            h_emit.view(-1, 1, self.hidden_dim),
            h_capture.view(-1, 1, self.hidden_dim)
        ], dim=1)
        h = self.block(h)
        h = self.head(h[:, 0])
        return h


@unique
class NeRGVariant(Enum):
    Emit = "Emit"
    Capture = "Capture"
    EmitCapture = "EmitCapture"
    EmitCaptureSemantic = "EmitCaptureSemantic"
    EmitCaptureSemanticNoAccumulate = "EmitCaptureSemanticNoAccumulate"
    EmitCaptureSemanticFinetune = "EmitCaptureSemanticFinetune"
    EmitCaptureSemanticFinetuneNoAccumulate = "EmitCaptureSemanticFinetuneNoAccumulate"
    EmitSemantic = "EmitSemantic"
    EmitSemanticFinetune = "EmitSemanticFinetune"

    # experiments with attention
    EmitCaptureAttend = "EmitCaptureAttend"

    def __new__(cls, name):
        obj = object.__new__(cls)
        obj._value_ = name
        name_lower = name.lower()
        obj.emit = 'emit' in name_lower
        obj.capture = 'capture' in name_lower
        obj.semantic = 'semantic' in name_lower
        obj.emit_geofeat = 'finetune' in name_lower or name_lower in ['emit', 'capture', 'emitcapture', 'emitcaptureattend']
        obj.semantic_noaccum = 'noaccumulate' in name_lower
        obj.attend = 'attend' in name_lower
        return obj

    def __str__(self):
        data_dict = {
            'emit': self.emit,
            'capture': self.capture,
            'emit_geofeat': self.emit_geofeat,
            'semantic': self.semantic,
            'semantic_noaccum': self.semantic_noaccum,
            'attend': self.attend,
        }
        return f"{self.value}: {data_dict}"


"""
This class follows the format of pytorch-ngp's NeRFRenderer to facilitate interfacing with their GUI.
"""
class NeRGRenderer(nn.Module):
    def __init__(self,
                 model_nerf: NeRFRenderer,
                 variant: NeRGVariant=NeRGVariant.EmitCapture,
                 capture_type='linear',  # linear or vmf
                 min_near=0.2,
                 bound=1.0,
                 hidden_dim_net1=64,
                 hidden_dim_net2=64,
                 hidden_dim_attn=64,
                 encoder_grid_resolution_per_unit_bound=256,
                 use_deepgaze_style_networks=True,
                 use_gaze_smoothing=False,
                 gaze_smoothing_learn_sigma=False,
                 gaze_smoothing_sigma=0.001,
                 ):
        super().__init__()

        self.model_nerf = model_nerf
        self.geo_feat_dim = model_nerf.geo_feat_dim

        self.cuda_ray = self.model_nerf.cuda_ray
        self.min_near = min_near
        self.bound = bound
        aabb_train = torch.FloatTensor([-bound, -bound, -bound, bound, bound, bound])
        self.register_buffer('aabb_train', aabb_train)

        self.encoder_grid_resolution_per_unit_bound = encoder_grid_resolution_per_unit_bound

        # position and direction encoders
        self.encoder, encoder_dim_out = get_encoder(
            "hashgrid",
            num_levels=16,
            # desired_resolution=encoder_grid_resolution_per_unit_bound,
            desired_resolution=encoder_grid_resolution_per_unit_bound * bound,  # scale grid res with bound
        )
        self.encoder_dir, encoder_dir_dim_out = get_encoder("sphere_harmonics", degree=4)

        print(f"Made NeRGNetwork with geo_feat_dim {self.geo_feat_dim} encoder_dir_dim_out {encoder_dir_dim_out}")

        if use_deepgaze_style_networks:
            # deepgaze2e-style nets
            self.net1, self.net2 = make_nets_deepgaze2e(
                encoder_dim_out, hidden_dim_net1, self.geo_feat_dim, encoder_dir_dim_out, hidden_dim_net2)
        else:
            # simple nerf-style nets
            self.net1, self.net2 = make_nets_nerf(
                encoder_dim_out, hidden_dim_net1, self.geo_feat_dim, encoder_dir_dim_out, hidden_dim_net2)

        self.variant = variant

        self.capture_type = capture_type

        if not self.variant.attend:
            self.emit_act = nn.Identity()
            self.capture_act = nn.Sigmoid()

            if self.variant.emit:
                self.emit_head = nn.Linear(hidden_dim_net2, 1, bias=False)

            if self.variant.capture:
                if self.capture_type == 'linear':
                    self.capture_head = nn.Linear(hidden_dim_net2, 1, bias=False)

                elif self.capture_type == 'vmf':  # experimental variant with gaussian mixture instead of lin layer
                    self.capture_head = VMFHead(
                        dim_in=self.geo_feat_dim,
                    )

                else:
                    raise ValueError("NeRG Unsupported capture type")

        else:  # experimental variant with attention
            if not self.variant.emit or not self.variant.capture:
                raise ValueError("NeRG with Attention requires emit and capture")
            if not self.capture_type == 'linear':
                raise ValueError("NeRG with Attention requires linear capture")

            self.emit_act = nn.Identity()
            self.capture_act = nn.Identity()
            self.emit_head = nn.Linear(hidden_dim_net2, hidden_dim_attn, bias=False)
            self.capture_head = nn.Linear(hidden_dim_net2, hidden_dim_attn, bias=False)
            self.attend_head = AttendBlock(hidden_dim=hidden_dim_attn, num_heads=4)

        # sanity check
        if self.variant.emit and (not self.variant.emit_geofeat and not (self.variant.semantic or self.variant.semantic_noaccum)):
            raise ValueError("Bad NeRG config: emit model has no geometric features for emit.")

        # applies a smoothing kernel to gaze predictions based on proximity of the gaze rays (over a sphere)
        self.use_gaze_smoothing = use_gaze_smoothing
        self.gaze_smoothing_learn_sigma = gaze_smoothing_learn_sigma
        if use_gaze_smoothing:
            if gaze_smoothing_learn_sigma:
                self.gaze_smoothing_sigma = nn.Parameter(torch.Tensor([gaze_smoothing_sigma]))  # learnable
            else:
                self.gaze_smoothing_sigma = gaze_smoothing_sigma

        self.init_networks()

    def init_networks(self):
        # initialize emite/capture params near zero
        with torch.no_grad():
            if self.variant.emit:
                self.emit_head.weight.data.normal_(0, 0.01)

            if self.variant.capture and self.capture_type == 'linear':
                self.capture_head.weight.data.normal_(0, 0.01)

    def train(self, mode: bool = True):
        super().train(mode)
        self.model_nerf.eval()  # nerf module is always in eval mode
        return self

    def update_extra_state(self):
        # do nothing, but warn
        log_warn("NeRGRenderer call to update_extra_state() is ignored.")

    def reset_extra_state(self):
        # do nothing, but warn
        log_warn("NeRGRenderer call to reset_extra_state() is ignored.")

    def forward(self, rays_o, rays_d, rays_od, geofeat=None, perturb=False, dt_gamma=0):
        # predict gaze seen at rays_o towards rays_d emitted from rays_og towards -rays_d
        # when perspective is aligned, rays_og = rays_o

        if perturb:
            # nudge positions
            rand_amount = self.get_max_perturb_amount(dt_gamma)  # dt_gamma scales perturb size
            rays_o = rays_o + torch.randn_like(rays_o) * rand_amount
            rays_od = rays_od + torch.randn_like(rays_od) * rand_amount

        if self.variant.attend:
            # gaze emitted from a surface towards the observer
            h_emit = self.gaze_emit(rays_od, -rays_d, geofeat)
            # gaze captured by the observer towards the surface
            h_capture = self.gaze_capture(rays_o, rays_d)
            gaze_prob = self.attend_head(h_emit, h_capture)

        else:
            # gaze emitted from a surface towards the observer
            gaze_emit = self.gaze_emit(rays_od, -rays_d, geofeat) if self.variant.emit else 1.0
            # gaze captured by the observer towards the surface
            gaze_capture = self.gaze_capture(rays_o, rays_d) if self.variant.capture else 1.0
            # gaze equation
            gaze_prob = gaze_capture * gaze_emit

        # tinfo('gaze_capture', gaze_capture, verbose=True)
        # tinfo('gaze_emit', gaze_emit, verbose=True)

        return gaze_prob.squeeze(-1)

    def gaze_emit(self, x, d, geofeat=None):
        if self.variant.semantic:
            # geofeat are precomputed by the NeRF model via density-based accumulation
            if geofeat is None:
                raise ValueError("Semantic-NeRF requires geofeat to be provided.")

        elif self.variant.semantic_noaccum:
            # when no accumulate variant is used, nerf geofeat are not precomputed.
            # evaluate a single snapshot of geofeat at the position of the surface.
            with torch.no_grad():
                geofeat = self.model_nerf.density(x)['geo_feat']

        else:
            geofeat = 0

        if self.variant.emit_geofeat:
            h = self.encoder(x, bound=self.bound)
            h = self.net1(h)
            geofeat = geofeat + h

        d = self.encoder_dir(d)
        h = torch.cat([d, geofeat], dim=-1)  # geo + dir
        h = self.net2(h)

        h = self.emit_head(h)
        h = self.emit_act(h)

        return h

    def gaze_capture(self, x, d):
        h = self.encoder(x, bound=self.bound)
        h = self.net1(h)  # geo feats

        if self.capture_type == 'vmf':
            mu, kappa, mix_logits = self.capture_head(h)  # [K,3], [K], [K]

            logC = vmf_log_density_mixture(d, mu, kappa, mix_logits)  # [N]

            return torch.exp(logC)

        elif self.capture_type == 'linear':
            d = self.encoder_dir(d)
            h = torch.cat([d, h], dim=-1)  # geo + dir

            h = self.net2(h)
            h = self.capture_head(h)
            h = self.capture_act(h)

            return h

        else:
            raise NotImplementedError()

    def predict_gaze(self, rays_o, rays_d, depth=None, geofeat=None, max_steps=1024, perturb=False, dt_gamma=0, gaze_smoothing=False):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: image: [B, N, 3], depth: [B, N]

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        # if depth/geofeats are not provided by NeRF (during training) or when perspectives are not aligned
        if depth is None or (self.variant.semantic and geofeat is None):
            # estimate depth/geofeats from observer (gaze) perspective
            with torch.no_grad():
                nerf_outputs = self.model_nerf.run_cuda_depth_geofeat(
                    rays_o, rays_d, dt_gamma=dt_gamma,
                    perturb=perturb,
                    max_steps=max_steps,
                    return_geofeat=self.variant.semantic,
                )
                depth = nerf_outputs['depth']
                geofeat = nerf_outputs['geofeat'] if self.variant.semantic else None

        if geofeat is not None:
            geofeat = geofeat.view(-1, self.geo_feat_dim)

        # position of the visible surface
        rays_od = rays_o + rays_d * depth.view(-1, 1)  # from rays_o along rays_d by distance depth

        gaze_prob = self(rays_o, rays_d, rays_od, geofeat=geofeat, perturb=perturb, dt_gamma=dt_gamma)

        if self.use_gaze_smoothing and gaze_smoothing:
            # regularize by smoothing over nearby rays
            gaze_prob = unit_sphere_smoothing(rays_d, gaze_prob, sigma=self.gaze_smoothing_sigma)

        gaze_prob = gaze_prob.view(*prefix)
        depth_w = depth.view(*prefix)  # depth in world space from observer's perspective

        results = {}
        results['depth_w'] = depth_w
        results['depth'] = depth_w
        results['gaze_prob'] = gaze_prob

        return results

    # optimizer utils
    def get_params(self, lr):
        params = [
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.net1.parameters(), 'lr': lr},
            {'params': self.encoder_dir.parameters(), 'lr': lr},
            {'params': self.net2.parameters(), 'lr': lr},
        ]

        if self.variant.emit:
            params.append(
                {'params': self.emit_head.parameters(), 'lr': lr},
            )

        if self.variant.capture:
            params.append(
                {'params': self.capture_head.parameters(), 'lr': lr},
            )

        if self.variant.attend:
            params.append(
                {'params': self.attend_head.parameters(), 'lr': lr},
            )

        if self.use_gaze_smoothing and self.gaze_smoothing_learn_sigma:
            params.append(
                {"params": self.gaze_smoothing_sigma, 'lr': lr}
            )

        return params

    def get_max_perturb_amount(self, dt_gamma):
        # perturb distance relative to the size of the encoded "cube" in world coordinates
        perturb_bound_ratio = dt_gamma * 1e-4
        # perturb distance scaled by dt_gamma and scene bound
        perturb_dist = perturb_bound_ratio * self.bound
        return perturb_dist


class GazeWrapper:
    """
    visualizes gaze patterns using i) NeRG, ii) DeepGaze2e, iii) raw gaze rays (gaze probes).
    """
    def __init__(
            self,
            model_nerg: NeRGRenderer,
            device,
            load_raw_gaze_data=False,  # toggle to load raw gaze data (1-3 minutes to load) to visualize gaze probe KDE
            load_deepgaze=True,  # predict 2D salience with DeepGaze model
            gaze_occlusion=False,
            gaze_occlusion_falloff=0.5
    ):
        # NeRF models for the scene and gaze prediction
        self.model_nerg = model_nerg

        if model_nerg is None:
            raise ValueError("GazeWrapper is missing the NeRG model.")

        # toggles to show gaze from DeepGaze or raw gaze data instead of using NeRG
        self.use_deepgaze = False
        self.use_raw_gaze_data = False

        # raw gaze data is slow per frame and slow to set up; this can be disabled if we only want to see NeRG/Deepgaze
        self.load_raw_gaze_data = load_raw_gaze_data
        if load_raw_gaze_data:
            self.dataset = GazeDataset(
                # get K closest gaze rays
                device=None,  # does not matter since Dataloader is not used
                gaze_query_type=GAZE_QUERY_TYPE_COUNT,  # this has to be COUNT, else can get gaze_spheres with no rays
                gaze_query_count=256,  # desired number of gaze rays for COUNT query
                preload_gaze_probes=False
            )

        if load_deepgaze:
            self.deepgaze = DeepGazeWrapper(device)

        self.gaze_occlusion = gaze_occlusion
        self.gaze_occlusion_falloff = gaze_occlusion_falloff

    def compute_deepgaze(self, image):
        gaze_log_prob = self.deepgaze.compute_gaze(255 * image)  # deepgaze was trained with images in 0-255
        return gaze_log_prob

    def requires_nerf_geofeat(self, aligned_perspective):
        return aligned_perspective and not (self.load_raw_gaze_data and self.use_raw_gaze_data) and self.model_nerg.variant.semantic

    def compute_gaze(self, pose, rays_o, rays_d, depth_d, geofeat, device, aligned_perspective=False, perturb=False, dt_gamma=0):
        # get rays for the gaze perspective
        if aligned_perspective:
            # if nerf camera and gaze camera perspectives are aligned, no need to recompute direction towards surface.
            # ray origin and direction are the same; expected depth is already known from the NeRF render pass.
            rays_oo = rays_o
            rays_do = rays_d
            depth_do = depth_d

        else:
            # TODO: something is wrong here when aligned_perspective=False but cameras are still aligned
            # may be related to copying gaze/render cameras

            # observer and render camera perspectives are not aligned.
            # we must compute the position of the visible surface (project from ro towards rd by depth d)
            rays_oo = torch.from_numpy(pose[:3, 3]).to(device).view(1, -1, 3)  # observer position; pose last column
            rays_od = rays_o + rays_d * depth_d.view(1, -1, 1)  # object surface, position along rays_d at depth
            rays_do = rays_od - rays_oo  # vector from observer to surface
            depth_do = magnitude_t(rays_do)  # expected depth
            rays_do = rays_do / (depth_do.view(1, -1, 1) + 1e-6)  # normalized direction from observer to surface
            rays_do = rays_do.view(1, -1, 3)

        if self.load_raw_gaze_data and self.use_raw_gaze_data:  # compute gaze from raw gaze data
            rays_oo_np = rays_oo.detach().cpu().numpy().reshape(-1, 3).transpose()
            rays_oo_np = np.mean(rays_oo_np, axis=1, keepdims=True)

            rays_oo_np = pos_nerf2gaze(rays_oo_np).transpose()  # N,3; convert from NeRF to gaze data coordinates
            gaze_sphere: GazeProbe = self.dataset.get_gaze_probes_at(rays_oo_np)[0]

            rays_do = rays_do.detach().cpu().numpy().reshape(-1, 3).transpose()
            rays_do = dir_nerf2gaze(rays_do)  # 3,N

            gaze_log_prob = gaze_sphere.density.evaluate_log_prob_density(rays_do)
            gaze_log_prob = torch.from_numpy(gaze_log_prob).to(device)

            gaze_prob = torch.exp(gaze_log_prob)

        else:  # compute gaze with NeRG gaze prediction network
            rays_oo = rays_oo.expand(1, rays_do.shape[1], 3)

            # when gaze shadowing is enabled, it is better to disable perturb to have a more accurate estimate of depth
            # Note: disabled this for now
            perturb = perturb and not self.gaze_occlusion

            self.model_nerg.eval()

            # Note: when render and observer perspectives are aligned, there is no need to rerender depth from the
            # observer's perspective as depth is already known (equal to render perspective depth).
            # there should also be no gaze shadowing as the surface is always visible to the camera.
            depth_g = depth_d if aligned_perspective else None
            geofeat = geofeat if aligned_perspective else None
            outputs_gaze = self.model_nerg.predict_gaze(
                rays_oo, rays_do, depth=depth_g, geofeat=geofeat, perturb=perturb, dt_gamma=dt_gamma
            )

            gaze_prob = outputs_gaze["gaze_prob"]

            if not aligned_perspective and self.gaze_occlusion:
                depth_do_actual = outputs_gaze["depth_w"]

                # test if depth is obscured by comparing expected and actual depths from gaze perspective.
                # when obscured, depth_do > depth_do_actual and the amount controls shadowing.
                # large depth delta produces noticeable shadowing
                depth_delta = depth_do.view(*depth_do_actual.shape) - depth_do_actual
                # allow some depth error which scales with dt_gamma
                depth_delta = depth_delta - 10 * self.model_nerg.get_max_perturb_amount(dt_gamma)

                # scale by falloff parameter and clip the shadowing factor
                shadowing_factor = depth_delta / (self.gaze_occlusion_falloff + 1e-4)
                shadowing_factor = torch.clip(shadowing_factor, 0.0, 1.0)

                gaze_prob = lerp(gaze_prob, 0.0, shadowing_factor)

        return gaze_prob
