#!/bin/bash
# Install package for the Kaggle

echo "--- Start install package ---"

pip install pip3-autoremove

pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu128

pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

pip install transformers==4.56.2

pip install --no-deps trl==0.22.2

echo "--- Installed successfully ---"