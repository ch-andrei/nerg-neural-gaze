# Neural Radiance and Gaze Fields (NeRGs)

[Repo under construction]

## Contents

This repository contains a Pytorch implementation for Neural Radiance and Gaze Fields (NeRGs), as described in:

<i> Andrei Chubarau, Yinan Wang, James J. Clark.
Neural Radiance and Gaze Fields for Visual Attention Modeling in 3D Environments (2025).
<a href="https://arxiv.org/abs/2503.07828">Arxiv link</a>.
</i>

Please cite as:

```
@article{nerg2025,
  title={Neural Radiance and Gaze Fields for Visual Attention Modeling in 3D Environments}, 
  author={Andrei Chubarau and Yinan Wang and James J. Clark},
  year={2025},
  eprint={2503.07828},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2503.07828}, 
}
```

## Paper Abstract

We introduce Neural Radiance and Gaze Fields (NeRGs), a novel approach for representing visual 
attention in complex environments. Much like how Neural Radiance Fields (NeRFs) perform novel 
view synthesis, NeRGs reconstruct gaze patterns from arbitrary viewpoints, implicitly mapping 
visual attention to 3D surfaces. We achieve this by augmenting a standard NeRF with an 
additional network that models local egocentric gaze probability density, conditioned on scene 
geometry and observer position. The output of a NeRG is a rendered view of the scene alongside 
a pixel-wise salience map representing the conditional probability that a given observer 
fixates on visible surfaces. Unlike prior methods, our system is lightweight and enables 
visualization of gaze fields at interactive framerates. Moreover, NeRGs allow the observer 
perspective to be decoupled from the rendering camera and correctly account for gaze occlusion 
due to intervening geometry. We demonstrate the effectiveness of NeRGs using head pose from 
skeleton tracking as a proxy for gaze, employing our proposed gaze probes to aggregate noisy 
rays into robust probability density targets for supervision.
