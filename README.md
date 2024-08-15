# Adaptive Russian Roulette and Path Space Filtering using Hash Caches and Neural Networks
### About
This repository contains the implementation of my master thesis in the `Falcor` engine from NVIDIA. It applies the `Hash Cache` data structure proposed by [Binder et al. \[2019\]](https://arxiv.org/abs/1902.05942) to store outgoing and incident radiance estimates in world space. Furthermore, a `Neural Network` as proposed by [Müller et al. \[2021\]](https://research.nvidia.com/publication/2021-06_real-time-neural-radiance-caching-path-tracing) is used to estimate the same quantities and compete with the `Hash Cache`. The `Neural Network` uses adaptions of the `Frequency Encoding` [\[Mildenhall et al. 2020\]](https://arxiv.org/abs/2003.08934), the `Spherical Harmonics Encoding` [\[Verbin et al. 2022\]](https://arxiv.org/abs/2112.03907), and the `Multiresolution Hash Encoding` [\[Müller et al. 2022\]](https://arxiv.org/abs/2201.05989).

The `Hash Caches` and the `Neural Caches` provide the global and local radiance estimate required for `Adjoint-Driven Russian Roulette` [\[Vorba and Křivánek 2016\]](https://dl.acm.org/doi/10.1145/2897824.2925912) to improve the path termination decisions of `Russian Roulette`. `Path Space Filtering` heavily relies on radiance estimates and the strategy for terminating paths. It can significantly speed up rendering at the cost of bias. The radiance estimates can be supplied by the `Hash Caches` and `Neural Caches`. Paths are terminated either by a heuristic based on the accumulated roughness of surfaces hit by the path or by `Adjoint-Driven Russian Roulette`.

### Adapted slang
This project requires an adapted version of `slang` which can be found at <https://github.com/itzMatze/slang>.
- clone the `slang` repository somewhere on your machine
- place a symlink under `external/packman/` with the name `slang` that points to the top level of the adapted `slang` repository

The new compilation procedure uses the `glslc` command line tool; so, it should be installed. Unfortunately, this compilation procedure will most likely not work on windows. However, it should only require very small adaptions to make it work.

The original readme of the `Falcor` project can be found [here](README_Falcor.md).
