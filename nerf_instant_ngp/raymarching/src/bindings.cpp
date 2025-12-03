#include <torch/extension.h>

#include "raymarching.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // utils
    m.def("packbits", &packbits, "packbits (CUDA)");
    m.def("near_far_from_aabb", &near_far_from_aabb, "near_far_from_aabb (CUDA)");
    m.def("sph_from_ray", &sph_from_ray, "sph_from_ray (CUDA)");
    m.def("morton3D", &morton3D, "morton3D (CUDA)");
    m.def("morton3D_invert", &morton3D_invert, "morton3D_invert (CUDA)");
    // train
    m.def("march_rays_train", &march_rays_train, "march_rays_train (CUDA)");
    m.def("composite_rays_train_forward", &composite_rays_train_forward, "composite_rays_train_forward (CUDA)");
    m.def("composite_rays_train_backward", &composite_rays_train_backward, "composite_rays_train_backward (CUDA)");
    m.def("composite_rays_train_forward_geofeat", &composite_rays_train_forward_geofeat, "composite_rays_train_forward_geofeat (CUDA)");
    m.def("composite_rays_train_backward_geofeat", &composite_rays_train_backward_geofeat, "composite_rays_train_backward_geofeat (CUDA)");
    // infer
    m.def("march_rays", &march_rays, "march rays (CUDA)");
    m.def("composite_rays", &composite_rays, "composite rays (CUDA)");
    m.def("composite_rays_rgb_geofeat", &composite_rays_rgb_geofeat, "composite rays (CUDA)");
    m.def("composite_rays_geofeat", &composite_rays_geofeat, "composite rays geofeat (CUDA)");
    m.def("composite_rays_depth", &composite_rays_depth, "composite rays depth (CUDA)");  // composite depth only
}