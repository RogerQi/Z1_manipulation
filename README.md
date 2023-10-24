# Z1_manipulation
Code for simulation envs and policy training on Z1

## Set up the environment (tested on volcan)

### Install Anaconda

Follow the instruction to install anaconda [here](https://www.anaconda.com/download).

Follow the instruction [here](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community) to set up Mamba, a fast environment solver for conda.

```
conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

Note: technically, the mamba solver should behave the same as the default solver. However, there have been cases where dependencies
can not be properly set up with the default mamba solver. The following instructions have **only** been tested on mamba solver.

### Setup conda

```bash
conda create -y -n b1 python=3.8 && conda activate b1
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Setup IsaacGym

Since IsaacGym package is an evolving package with rapidly changing APIs, for compatibility, we attach a version of isaacgym in the repo.

```bash
cd isaacgym/python
pip install .
```

Note: for some unknown reason, sometimes `gymtorch` headers are **NOT** copied to site-packages. We need to double check its existence

```bash
ls ~/anaconda3/envs/b1/lib/python3.8/site-packages/isaacgym/_bindings/src/gymtorch
```

If it's not there, one quick but dirty way to fix it is to manually copy the missing headers. Make sure you are in ```/B1_manipulation/isaacgym/python``` before running this command

```bash
cp -r ./isaacgym/_bindings/src ~/anaconda3/envs/b1/lib/python3.8/site-packages/isaacgym/_bindings/
```
