
import torch.nn.functional as f

from copy import deepcopy

import dearpygui.dearpygui as dpg
from scipy.spatial.transform import Rotation as R

from common.vector_math import lerp
from common.viridis_cm import apply_viridis_cm
from nerg.gaze_prediction import GazeWrapper, NeRGRenderer

from nerf_instant_ngp.nerf.utils import *


class OrbitCamera:
    def __init__(self, W, H, r=2, fovy=90):
        self.W = W
        self.H = H
        self.radius = r  # camera distance from center
        self.fovy = fovy  # in degree
        self.center = np.array([0, 0, 0], dtype=np.float32)  # look at this point
        self.rot = R.from_quat(
            [1, 0, 0, 0])  # init camera matrix: [[1, 0, 0], [0, -1, 0], [0, 0, 1]] (to suit ngp convention)
        self.up = np.array([0, 1, 0], dtype=np.float32)  # need to be normalized!

    # pose
    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4, dtype=np.float32)
        res[2, 3] -= self.radius
        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res

    # intrinsics
    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(np.radians(self.fovy) / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2])

    def orbit(self, dx, dy):
        # rotate along camera up/side axis!
        side = self.rot.as_matrix()[:3, 0]  # why this is side --> ? # already normalized.
        rotvec_x = self.up * np.radians(-0.1 * dx)
        rotvec_y = side * np.radians(-0.1 * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([dx, dy, dz])


"""
in addition to NeRF GUI controls, NeRGGUI also includes NeRG overlay controls for displaying gaze.
"""
class NeRFGUI:
    def __init__(self, opt, trainer, train_loader=None, debug=True, use_gaze=False):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)
        self.debug = debug
        self.bg_color = torch.zeros(3, dtype=torch.float32)  # default white bg
        self.training = False
        self.step = 0  # training step

        self.trainer = trainer
        self.train_loader = train_loader
        if train_loader is not None:
            self.trainer.error_map = train_loader._data.error_map

        self.render_buffer = np.zeros((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True  # camera moved, should reset accumulation
        self.spp = 1  # sample per pixel
        self.mode = 'image'  # choose from ['image', 'depth']

        self.dynamic_resolution = False
        self.downscale = 1
        self.train_steps = 16

        self.use_gaze = use_gaze
        self.gaze_gui = None
        if use_gaze:
            self.gaze_gui = GazeGUI(self)

        dpg.create_context()
        self.register_dpg()
        self.test_step()

    def __del__(self):
        dpg.destroy_context()

    def train_step(self):
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()

        outputs = self.trainer.train_gui(self.train_loader, step=self.train_steps)

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        self.step += self.train_steps
        self.need_update = True

        dpg.set_value("_log_train_time", f'{t:.4f}ms ({int(1000 / t)} FPS)')
        dpg.set_value("_log_train_log",
                      f'step = {self.step: 5d} (+{self.train_steps: 2d}), loss = {outputs["loss"]:.4f}, lr = {outputs["lr"]:.5f}')

        # dynamic train steps
        # max allowed train time per-frame is 500 ms
        full_t = t / self.train_steps * 16
        train_steps = min(16, max(4, int(16 * 500 / full_t)))
        if train_steps > self.train_steps * 1.2 or train_steps < self.train_steps * 0.8:
            self.train_steps = train_steps

    def prepare_buffer(self, outputs):
        if self.mode == 'image':
            return outputs['image']
        elif self.mode == 'depth':
            return np.expand_dims(outputs['depth'], -1).repeat(3, -1)
        elif self.mode == 'depth_w':
            depth_w = outputs['depth_w']
            depth_w -= depth_w.min()
            depth_w /= depth_w.max() + 1e-6
            depth_w = depth_w.detach().cpu().numpy()
            return np.expand_dims(depth_w, -1).repeat(3, -1)
        else:
            raise ValueError()

    def test_step(self):
        if self.need_update or self.spp < self.opt.max_spp:

            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()

            if self.use_gaze:
                # use Gaze GUI to render the scene: NeRF + gaze from gaze wrapper
                aligned_perspective = (
                    self.gaze_gui.aligned_perspective or
                    np.sum(np.abs((self.gaze_gui.cam_gaze.pose - self.cam.pose))) < 0.025  # tolerance for alignment
                )
                outputs = self.gaze_gui.test_gui(
                    self.cam.pose, self.cam.intrinsics, self.W, self.H, self.bg_color, self.spp, self.downscale,
                    aligned_perspective=aligned_perspective, dt_gamma=self.opt.dt_gamma
                )
            else:
                # use NeRGTrainer of NeRFTrainer to render the scene
                outputs = self.trainer.test_gui(
                    self.cam.pose, self.cam.intrinsics, self.W, self.H, self.bg_color, self.spp, self.downscale)

            ender.record()
            torch.cuda.synchronize()
            t = starter.elapsed_time(ender)

            # update dynamic resolution
            if self.dynamic_resolution:
                # max allowed infer time per-frame is 200 ms
                full_t = t / (self.downscale ** 2)
                downscale = min(1, max(1 / 4, math.sqrt(200 / full_t)))
                if downscale > self.downscale * 1.2 or downscale < self.downscale * 0.8:
                    self.downscale = downscale

            if self.need_update:
                self.render_buffer = self.prepare_buffer(outputs)
                self.spp = 1
                self.need_update = False
            else:
                self.render_buffer = (self.render_buffer * self.spp + self.prepare_buffer(outputs)) / (self.spp + 1)
                self.spp += 1

            # print('up dot', np.dot(self.cam.up.numpy(), np.array([0, 1, 0])))

            dpg.set_value("_log_infer_time", f'{t:.4f}ms ({int(1000 / t)} FPS)')
            dpg.set_value("_log_resolution", f'{int(self.downscale * self.W)}x{int(self.downscale * self.H)}')
            dpg.set_value("_log_spp", self.spp)
            dpg.set_value("_texture", self.render_buffer)

    def register_dpg(self):

        ### register texture

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.W, self.H, self.render_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")

        ### register window

        # the rendered image, as the primary window
        with dpg.window(tag="_primary_window", width=self.W, height=self.H):

            # add the texture
            dpg.add_image("_texture")

        dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(label="Control", tag="_control_window", width=400, height=300):

            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # time
            if not self.opt.test:
                with dpg.group(horizontal=True):
                    dpg.add_text("Train time: ")
                    dpg.add_text("no data", tag="_log_train_time")

            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")

            with dpg.group(horizontal=True):
                dpg.add_text("SPP: ")
                dpg.add_text("1", tag="_log_spp")

            # train button
            if not self.opt.test:
                with dpg.collapsing_header(label="Train", default_open=True):

                    # train / stop
                    with dpg.group(horizontal=True):
                        dpg.add_text("Train: ")

                        def callback_train(sender, app_data):
                            if self.training:
                                self.training = False
                                dpg.configure_item("_button_train", label="start")
                            else:
                                self.training = True
                                dpg.configure_item("_button_train", label="stop")

                        dpg.add_button(label="start", tag="_button_train", callback=callback_train)
                        dpg.bind_item_theme("_button_train", theme_button)

                        def callback_reset(sender, app_data):
                            @torch.no_grad()
                            def weight_reset(m: nn.Module):
                                reset_parameters = getattr(m, "reset_parameters", None)
                                if callable(reset_parameters):
                                    m.reset_parameters()

                            self.trainer.model.apply(fn=weight_reset)
                            self.trainer.model.reset_extra_state()  # for cuda_ray density_grid and step_counter
                            self.need_update = True

                        dpg.add_button(label="reset", tag="_button_reset", callback=callback_reset)
                        dpg.bind_item_theme("_button_reset", theme_button)

                    # save ckpt
                    with dpg.group(horizontal=True):
                        dpg.add_text("Checkpoint: ")

                        def callback_save(sender, app_data):
                            self.trainer.save_checkpoint(full=True, best=False)
                            dpg.set_value("_log_ckpt",
                                          "saved " + os.path.basename(self.trainer.stats["checkpoints"][-1]))
                            self.trainer.epoch += 1  # use epoch to indicate different calls.

                        dpg.add_button(label="save", tag="_button_save", callback=callback_save)
                        dpg.bind_item_theme("_button_save", theme_button)

                        dpg.add_text("", tag="_log_ckpt")

                    # save mesh
                    with dpg.group(horizontal=True):
                        dpg.add_text("Marching Cubes: ")

                        def callback_mesh(sender, app_data):
                            self.trainer.save_mesh(resolution=256, threshold=10)
                            dpg.set_value("_log_mesh", "saved " + f'{self.trainer.name}_{self.trainer.epoch}.ply')
                            self.trainer.epoch += 1  # use epoch to indicate different calls.

                        dpg.add_button(label="mesh", tag="_button_mesh", callback=callback_mesh)
                        dpg.bind_item_theme("_button_mesh", theme_button)

                        dpg.add_text("", tag="_log_mesh")

                    with dpg.group(horizontal=True):
                        dpg.add_text("", tag="_log_train_log")

            # rendering options
            with dpg.collapsing_header(label="Options", default_open=True):
                # dynamic rendering resolution
                with dpg.group(horizontal=True):

                    def callback_set_dynamic_resolution(sender, app_data):
                        if self.dynamic_resolution:
                            self.dynamic_resolution = False
                            self.downscale = 1
                        else:
                            self.dynamic_resolution = True
                        self.need_update = True

                    dpg.add_checkbox(label="dynamic resolution", default_value=self.dynamic_resolution,
                                     callback=callback_set_dynamic_resolution)
                    dpg.add_text(f"{self.W}x{self.H}", tag="_log_resolution")

                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True

                dpg.add_combo(('image', 'depth', 'depth_w'), label='mode', default_value=self.mode, callback=callback_change_mode)

                # bg_color picker
                def callback_change_bg(sender, app_data):
                    self.bg_color = torch.tensor(app_data[:3], dtype=torch.float32)  # only need RGB in [0, 1]
                    self.need_update = True

                dpg.add_color_edit((0, 0, 0), label="Background Color", width=200, tag="_color_editor", no_alpha=True,
                                   callback=callback_change_bg)

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = app_data
                    self.need_update = True

                dpg.add_slider_int(label="FoV (vertical)", min_value=1, max_value=120, format="%d deg",
                                   default_value=self.cam.fovy, callback=callback_set_fovy)

                # dt_gamma slider
                def callback_set_dt_gamma(sender, app_data):
                    self.opt.dt_gamma = app_data
                    self.need_update = True

                dpg.add_slider_float(label="dt_gamma", min_value=0, max_value=0.1, format="%.5f",
                                     default_value=self.opt.dt_gamma, callback=callback_set_dt_gamma)

                # max_steps slider
                def callback_set_max_steps(sender, app_data):
                    self.opt.max_steps = app_data
                    self.need_update = True

                dpg.add_slider_int(label="max steps", min_value=1, max_value=1024, format="%d",
                                   default_value=self.opt.max_steps, callback=callback_set_max_steps)

                # aabb slider
                def callback_set_aabb(sender, app_data, user_data):
                    # user_data is the dimension for aabb (xmin, ymin, zmin, xmax, ymax, zmax)
                    if isinstance(self.trainer.model, NeRGRenderer):
                        self.trainer.model.aabb_train[user_data] = app_data  # nerg only has aabb_train
                        self.trainer.model_nerf.aabb_infer[user_data] = app_data
                    else:
                        self.trainer.model.aabb_infer[user_data] = app_data

                    # also change train aabb ? [better not...]
                    # self.trainer.model.aabb_train[user_data] = app_data

                    self.need_update = True

                dpg.add_separator()
                dpg.add_text("Axis-aligned bounding box:")

                with dpg.group(horizontal=True):
                    dpg.add_slider_float(label="x", width=150, min_value=-self.opt.bound, max_value=0, format="%.2f",
                                         default_value=-self.opt.bound, callback=callback_set_aabb, user_data=0)
                    dpg.add_slider_float(label="", width=150, min_value=0, max_value=self.opt.bound, format="%.2f",
                                         default_value=self.opt.bound, callback=callback_set_aabb, user_data=3)

                with dpg.group(horizontal=True):
                    dpg.add_slider_float(label="y", width=150, min_value=-self.opt.bound, max_value=0, format="%.2f",
                                         default_value=-self.opt.bound, callback=callback_set_aabb, user_data=1)
                    dpg.add_slider_float(label="", width=150, min_value=0, max_value=self.opt.bound, format="%.2f",
                                         default_value=self.opt.bound, callback=callback_set_aabb, user_data=4)

                with dpg.group(horizontal=True):
                    dpg.add_slider_float(label="z", width=150, min_value=-self.opt.bound, max_value=0, format="%.2f",
                                         default_value=-self.opt.bound, callback=callback_set_aabb, user_data=2)
                    dpg.add_slider_float(label="", width=150, min_value=0, max_value=self.opt.bound, format="%.2f",
                                         default_value=self.opt.bound, callback=callback_set_aabb, user_data=5)

                if self.use_gaze:
                    self.gaze_gui.register_dpg()

            # debug info
            if self.debug:
                with dpg.collapsing_header(label="Debug"):
                    # pose
                    dpg.add_separator()
                    dpg.add_text("Camera Pose:")
                    dpg.add_text(str(self.cam.pose), tag="_log_pose")

        ### register camera handler

        def callback_camera_drag_rotate(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            if self.use_gaze:
                self.gaze_gui.update_gaze_cam()

            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))

        def callback_camera_wheel_scale(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            if self.use_gaze:
                self.gaze_gui.update_gaze_cam()
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            if self.use_gaze:
                self.gaze_gui.update_gaze_cam()
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))

        def on_move_key(sender, app_data):
            print(f"on_move_key, dpg.is_key_down(dpg.mvKey_Control) {dpg.is_key_down(dpg.mvKey_Control)}")

            if not dpg.is_key_down(dpg.mvKey_Control):
                return

            print("on_move_key exec")

            dx = 0
            dy = 0
            dz = 0
            amount = 20  # corresponds to pixels, i.e., move amount
            if app_data == dpg.mvKey_Q:
                dy = -amount
            elif app_data == dpg.mvKey_E:
                dy = amount
            elif app_data == dpg.mvKey_W:
                dz = -amount
            elif app_data == dpg.mvKey_S:
                dz = amount
            elif app_data == dpg.mvKey_A:
                dx = amount
            elif app_data == dpg.mvKey_D:
                dx = -amount

            self.cam.pan(dx, dy, dz)
            if self.use_gaze:
                self.gaze_gui.update_gaze_cam()
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))

        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate)
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan)

            # control keys
            dpg.add_key_press_handler(dpg.mvKey_Q, callback=on_move_key)
            dpg.add_key_press_handler(dpg.mvKey_W, callback=on_move_key)
            dpg.add_key_press_handler(dpg.mvKey_E, callback=on_move_key)
            dpg.add_key_press_handler(dpg.mvKey_A, callback=on_move_key)
            dpg.add_key_press_handler(dpg.mvKey_S, callback=on_move_key)
            dpg.add_key_press_handler(dpg.mvKey_D, callback=on_move_key)

        if self.use_gaze:
            self.gaze_gui.register_dpg_keys()

        dpg.create_viewport(title='torch-ngp', width=self.W, height=self.H, resizable=False)

        # TODO: seems dearpygui doesn't support resizing texture...
        # def callback_resize(sender, app_data):
        #     self.W = app_data[0]
        #     self.H = app_data[1]
        #     # how to reload texture ???

        # dpg.set_viewport_resize_callback(callback_resize)

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        # dpg.show_metrics()

        dpg.show_viewport()

    def render(self):
        while dpg.is_dearpygui_running():
            # update texture every frame
            if self.training:
                self.train_step()
            self.test_step()
            dpg.render_dearpygui_frame()


class GazeGUI:
    def __init__(
            self,
            main_gui: NeRFGUI,
    ):
        self.main_gui = main_gui
        self.show_gaze_overlay = False  # start with gaze off
        self.cam_gaze = OrbitCamera(main_gui.cam.W, main_gui.cam.H, main_gui.cam.radius, main_gui.cam.fovy)
        self.aligned_perspective = True
        self.gaze_alpha = 0.75

        self.opt_max_spp = self.main_gui.opt.max_spp
        self.device = self.main_gui.trainer.device
        self.fp16 = self.main_gui.trainer.fp16

        self.gaze_wrapper = GazeWrapper(
            model_nerg=self.main_gui.trainer.model,
            device=self.device,
        )

        self.update_gaze_cam()

    def update_gaze_cam(self, reset_render_cam=False):

        def match_cam(cam1: OrbitCamera, cam2: OrbitCamera):
            # match render camera params to gaze camera
            cam1.radius = cam2.radius
            cam1.fovy = cam2.fovy
            cam1.center = cam2.center.copy()
            cam1.rot = deepcopy(cam2.rot)
            cam1.up = cam2.up.copy()

        if reset_render_cam:
            # match render camera params to gaze camera
            match_cam(self.main_gui.cam, self.cam_gaze)
            self.main_gui.need_update = True

        if self.aligned_perspective:
            # match gaze camera params to render camera
            match_cam(self.cam_gaze, self.main_gui.cam)
            self.main_gui.need_update = True

    def test_gui(self, pose, intrinsics, W, H, bg_color=None, spp=1, downscale=1, aligned_perspective=True, dt_gamma=0):
        # render resolution (may need downscale to for better frame rate)
        rH = int(H * downscale)
        rW = int(W * downscale)
        intrinsics_ds = intrinsics * downscale

        pose = torch.from_numpy(pose).unsqueeze(0).to(self.device)
        rays = get_rays(pose, intrinsics_ds, rH, rW, -1)
        data = {
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'H': rH,
            'W': rW,
        }

        self.main_gui.trainer.model.eval()

        if self.main_gui.trainer.ema is not None:
            self.main_gui.trainer.ema.store()
            self.main_gui.trainer.ema.copy_to()

        perturb = False if spp == 1 else spp

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.fp16):
                nerf_return_geofeat = self.show_gaze_overlay and self.gaze_wrapper.requires_nerf_geofeat(aligned_perspective)
                # get the NeRF rendered image (shape 1xHxW) and optionally geofeats
                outputs = self.main_gui.trainer.render_nerf(
                    data, bg_color=bg_color, perturb=perturb, return_geofeat=nerf_return_geofeat)

                if nerf_return_geofeat:
                    pred_rgb, pred_depth, pred_depth_w, pred_geofeat = outputs
                else:
                    pred_rgb, pred_depth, pred_depth_w = outputs
                    pred_geofeat = None

                # get gaze probability
                gaze_prob = None
                if self.show_gaze_overlay:  # gaze overlay can be disabled
                    gaze_prob = self.get_gaze_prob(
                        pred_rgb, data, pred_depth_w, pred_geofeat,
                        aligned_perspective=aligned_perspective, perturb=perturb, dt_gamma=dt_gamma
                    )
                    gaze_prob = gaze_prob.view(-1, rH, rW)

        if self.main_gui.trainer.ema is not None:
            self.main_gui.trainer.ema.restore()

        # interpolation to the original resolution
        if downscale != 1:
            # TODO: have to permute twice with torch...
            pred_rgb = f.interpolate(pred_rgb.permute(0, 3, 1, 2), size=(H, W), mode='nearest').permute(0, 2, 3, 1).contiguous()
            pred_depth = f.interpolate(pred_depth.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)
            pred_depth_w = f.interpolate(pred_depth_w.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)
            gaze_prob = f.interpolate(gaze_prob.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1) if self.show_gaze_overlay else None

        pred_rgb = pred_rgb[0].detach().cpu().numpy()
        pred_depth = pred_depth[0].detach().cpu().numpy()
        pred_depth_w = pred_depth_w[0].detach()  # this needs to be a torch tensor for nerg-gui
        gaze_prob = gaze_prob[0].detach().cpu().numpy() if self.show_gaze_overlay else None

        gaze_color = None
        if self.show_gaze_overlay:
            # # TODO: should we correct for pixel solid angle? this acts as center bias
            # if not self.gaze_wrapper.use_deepgaze:
            #     # # given camera params: H,W, fx,fy,cx,cy; dirs[u,v]=unit ray; model outputs per-steradian density p(d)
            #     fx, fy, cx, cy = intrinsics
            #     u = np.arange(W)  # pixel columns
            #     v = np.arange(H)  # pixel rows
            #     u_grid, v_grid = np.meshgrid(u, v)  # shapes [H,W], [H,W]
            #     x = (u_grid - cx) / fx
            #     y = (v_grid - cy) / fy
            #     Omega = 1.0 / (fx * fy) / np.power(1.0 + x * x + y * y, 1.5)  # [H,W]
            #     gaze_prob = gaze_prob * Omega.reshape(-1, H, W)  # pixel mass

            # normalize gaze_prob to 0-1 and apply color map
            gaze_prob -= gaze_prob.min()
            gaze_prob /= gaze_prob.max()

            if self.main_gui.opt.gaze_cmap == "viridis":
                gaze_color = apply_viridis_cm(np.nan_to_num(gaze_prob, posinf=1.0, nan=1.0))

            else:
                gaze_color = np.zeros((*gaze_prob.shape, 3), float)
                gaze_color[..., :] = gaze_prob  # put to all channels

            pred_rgb = lerp(pred_rgb, gaze_color, self.gaze_alpha)

        outputs = {
            # tensors
            'data': data,
            'depth_w': pred_depth_w,
            # ndarrays
            'image': pred_rgb,
            'depth': pred_depth,
        }

        if self.show_gaze_overlay:
            outputs['gaze_prob'] = gaze_prob
            outputs['gaze_color'] = gaze_color

        return outputs

    def get_gaze_prob(self, image, data, depth_w, geofeat=None, aligned_perspective=True, perturb=False, dt_gamma=0):
        rays_o = data['rays_o']
        rays_d = data['rays_d']

        if self.gaze_wrapper.use_deepgaze:
            # use deepgaze to estimate saliency over the rendered image
            gaze_log_prob = self.gaze_wrapper.compute_deepgaze(image)
            gaze_prob = torch.exp(gaze_log_prob)

        else:
            # use nerg gaze prediction network to estimate gaze along rays from observer position
            gaze_prob = self.gaze_wrapper.compute_gaze(
                self.cam_gaze.pose, rays_o, rays_d, depth_w, geofeat, self.device,
                aligned_perspective=aligned_perspective, perturb=perturb, dt_gamma=dt_gamma)

        return gaze_prob

    def register_dpg(self):
        dpg.add_separator()
        dpg.add_text("Gaze prediction controls:")

        def callback_set_show_gaze(sender, app_data):
            self.show_gaze_overlay = app_data
            self.main_gui.need_update = True

        dpg.add_checkbox(label="Show gaze", default_value=self.show_gaze_overlay, callback=callback_set_show_gaze)

        def callback_set_gaze_aligned_perspective(sender, app_data):
            reset_render_cam = not self.aligned_perspective and app_data
            self.aligned_perspective = app_data
            self.main_gui.need_update = True
            self.update_gaze_cam(reset_render_cam)

        dpg.add_checkbox(label="Align Observer to Render camera", default_value=self.aligned_perspective,
                         callback=callback_set_gaze_aligned_perspective)

        def callback_set_gaze_occlusion(sender, app_data):
            self.gaze_wrapper.gaze_occlusion = app_data
            self.main_gui.need_update = True

        dpg.add_checkbox(label="Use gaze occlusion", default_value=self.gaze_wrapper.gaze_occlusion,
                         callback=callback_set_gaze_occlusion)

        def callback_set_use_deepgaze(sender, app_data):
            self.gaze_wrapper.use_deepgaze = app_data
            self.main_gui.need_update = True

            # limit spp to 1, because deepgaze is slow to compute
            if self.gaze_wrapper.use_deepgaze:
                self.main_gui.opt.max_spp = 1
            else:
                self.main_gui.opt.max_spp = self.opt_max_spp

        dpg.add_checkbox(label="Use DeepGaze", default_value=self.gaze_wrapper.use_deepgaze,
                         callback=callback_set_use_deepgaze)

        def callback_set_show_raw_gaze(sender, app_data):
            self.gaze_wrapper.use_raw_gaze_data = app_data
            self.main_gui.need_update = True

            # limit spp to 1, because raw gaze data is slow to compute
            if self.gaze_wrapper.load_raw_gaze_data:
                if self.gaze_wrapper.use_raw_gaze_data:
                    self.main_gui.opt.max_spp = 1
                else:
                    self.main_gui.opt.max_spp = self.opt_max_spp

        dpg.add_checkbox(label="Use gaze probes", default_value=self.gaze_wrapper.use_raw_gaze_data,
                         callback=callback_set_show_raw_gaze)

        def callback_set_gaze_alpha(sender, app_data):
            self.gaze_alpha = app_data
            self.main_gui.need_update = True

        dpg.add_slider_float(label="Gaze alpha", min_value=0., max_value=1.0, format="%.2f",
                             default_value=self.gaze_alpha, callback=callback_set_gaze_alpha, user_data=0)

        def callback_set_gaze_occlusion_falloff(sender, app_data):
            self.gaze_wrapper.gaze_occlusion_falloff = app_data
            self.main_gui.need_update = True

        dpg.add_slider_float(label="Occlusion falloff", min_value=0.01, max_value=5.0, format="%.3f",
                             default_value=self.gaze_wrapper.gaze_occlusion_falloff,
                             callback=callback_set_gaze_occlusion_falloff, user_data=0)

    def register_dpg_keys(self):
        def on_move_key_gaze(sender, app_data):

            print(f"on_move_key_gaze, dpg.is_key_down(dpg.mvKey_Alt) {dpg.is_key_down(dpg.mvKey_Alt)}, control_cam_gaze {self.aligned_perspective}")

            if not dpg.is_key_down(dpg.mvKey_Alt) or self.aligned_perspective:
                return

            print(f"on_move_key_gaze exec")

            dx = 0
            dy = 0
            dz = 0
            amount = 20  # corresponds to pixels, i.e., move amount
            if app_data == dpg.mvKey_Q:
                dy = -amount
            elif app_data == dpg.mvKey_E:
                dy = amount
            elif app_data == dpg.mvKey_W:
                dz = -amount
            elif app_data == dpg.mvKey_S:
                dz = amount
            elif app_data == dpg.mvKey_A:
                dx = amount
            elif app_data == dpg.mvKey_D:
                dx = -amount

            self.cam_gaze.pan(dx, dy, dz)
            self.main_gui.need_update = True

        with dpg.handler_registry():
            # control keys
            dpg.add_key_press_handler(dpg.mvKey_Q, callback=on_move_key_gaze)
            dpg.add_key_press_handler(dpg.mvKey_W, callback=on_move_key_gaze)
            dpg.add_key_press_handler(dpg.mvKey_E, callback=on_move_key_gaze)
            dpg.add_key_press_handler(dpg.mvKey_A, callback=on_move_key_gaze)
            dpg.add_key_press_handler(dpg.mvKey_S, callback=on_move_key_gaze)
            dpg.add_key_press_handler(dpg.mvKey_D, callback=on_move_key_gaze)
