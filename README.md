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

### Setup conda and PyTorch

```bash
conda create -y -n b1 python=3.8 && conda activate b1
conda install pytorch==2.1.0 torchvision==0.16.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Setup IsaacGym

Since IsaacGym package is an evolving package with rapidly changing APIs, for compatibility, we attach a version of isaacgym in the repo.

```bash
cd third_party/isaacgym/python
pip install .
```

Note: for some unknown reason, sometimes `gymtorch` headers are **NOT** copied to site-packages. We need to double check its existence

```bash
ls ~/anaconda3/envs/b1/lib/python3.8/site-packages/isaacgym/_bindings/src/gymtorch
```

If it's not there, one quick but dirty way to fix it is to manually copy the missing headers. Make sure you are in ```./third_party/isaacgym/python``` before running this command

```bash
cp -r ./isaacgym/_bindings/src ~/anaconda3/envs/b1/lib/python3.8/site-packages/isaacgym/_bindings/
```

You can test the installation by,

```bash
cd ./third_party/issacgym/python/examples
python joint_monkey.py
```

Note: if you see the following error,

```bash
ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory
```

You can do

```bash
export LD_LIBRARY_PATH=$HOME/anaconda3/envs/b1/lib:$LD_LIBRARY_PATH
```
