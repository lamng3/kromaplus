# kromaplus

## Setup

1. Spin up a RunPod instance

```
choose 1 RTX A4500 ($0.25/hr) on Axolotl Docker image.
```

2. Create conda env

```
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create --name kromaplus python=3.12 -y
source ~/.bashrc # or ~/.zshrc if you're using zsh
conda init
conda activate kromaplus
```

3. Install dependencies

```
cd kromaplus
pip install -r requirements.txt
pip install hf_transfer
pip install -e .
```

4. Push code to GitHub

```
<!-- generate SSH key (if not yet) -->
ssh-keygen -t ed25519 -C "your_email@example.com"

<!-- add public key to github -->
cat ~/.ssh/id_ed25519.pub

<!-- change repo remote to SSH -->
git remote set-url origin git@github.com:<username>/<repo_name>.git

<!-- test connection -->
ssh -T git@github.com
```

5. Play with the source in `main.py`

```
python main.py
```

6. Unit Test

```
pytest
```