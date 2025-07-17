# kromaplus

## Setup

1. Spin up a RunPod instance

2. Create conda env
```
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create --name kromaplus python=3.12 -y
source ~/.bashrc # or ~/.zshrc if you're using zsh
conda init
conda activate kromaplus
```