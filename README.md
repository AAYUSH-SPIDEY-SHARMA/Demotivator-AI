# README.md

# Demotivator AI Backend

Production-ready FastAPI backend for a ChatGPT-style demotivation service powered by vLLM.

## System Requirements

- Ubuntu 22.04 (WSL2 compatible)
- NVIDIA RTX 4060 or better (8+ GB VRAM)
- Python 3.10+
- CUDA 11.8 or 12.1

## Setup Instructions

### 1. Conda Environment Setup

```bash
# Create conda environment
conda create -n demotivator python=3.10
conda activate demotivator

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install backend dependencies
pip install -r requirements.txt