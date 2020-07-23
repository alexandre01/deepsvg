![pull_figure](imgs/pull_figure.png)

Paper: [arXiv](https://arxiv.org/abs/2007.11301) <br />
Code: [GitHub](https://github.com/alexandre01/deepsvg) <br />
Project page: [link](https://alexandre01.github.io/deepsvg)

## Introduction
Our model:
![architecture](imgs/architecture.png)

See [architecture.gif](imgs/architecture.gif) for an animated walk-through of our DeepSVG architecture.

## Demonstration
![gui](imgs/gui.gif)

## Interpolations
![latent_space](imgs/latent_space.gif)

- Betweeen a pair of icons:
<p align="center">
    <img alt="1765_47599" src="imgs/interpolations/1765_47599.gif">
    <img alt="8251_102098" src="imgs/interpolations/8251_102098.gif">
    <img alt="43800_93247" src="imgs/interpolations/43800_93247.gif">
</p>


- Between two user-drawn frames:
<p align="center">
    <img alt="bird" src="imgs/animations/bird.gif">
    <img alt="ship" src="imgs/animations/ship.gif">
    <img alt="foot" src="imgs/animations/foot.gif">
</p>


- Latent space algebra ("create" & "squarify"):
<p align="center">
    <img alt="baloon" src="imgs/latent_ops/baloon.gif">
    <img alt="bubbles" src="imgs/latent_ops/bubbles.gif">
    <img alt="drill" src="imgs/latent_ops/drill.gif">
</p>

## Citation
If you find this work useful in your research, please cite:
```
@misc{carlier2020deepsvg,
    title={DeepSVG: A Hierarchical Generative Network for Vector Graphics Animation},
    author={Alexandre Carlier and Martin Danelljan and Alexandre Alahi and Radu Timofte},
    year={2020},
    eprint={2007.11301},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```