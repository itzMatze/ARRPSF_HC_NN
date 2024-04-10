# Adaptive Russian Roulette using Neural Networks
The original readme of the `Falcor` project can be found [here](README_Falcor.md).
## Adapted slang
This project requires an adapted version of `slang` which can be found at <https://github.com/itzMatze/slang>.
- retrieve all dependencies of Falcor with packman
- clone the `slang` repository somewhere on your machine
- under `external/packman/` should be a symlink with the name `slang`
- let this symlink point to the top level of the adapted `slang` repository

The new compilation procedure uses the `glslc` command line tool; so, it should be installed. Unfortunately, this compilation procedure will most likely not work on windows. However, it should only require very small adaptions to make it work.

