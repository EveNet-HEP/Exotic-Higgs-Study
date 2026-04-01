# Exotic-Higgs-Study 

[![HF dataset](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-sm.svg)](https://huggingface.co/datasets/Avencast/EveNet-ExoticHiggs-H2a4b)
This repository contains the code and data for the Exotic Higgs Study, 
which investigates potential new physics phenomena in the Higgs boson sector.
## Installation
To set up the environment for this study, follow these steps:
```aiignore
git clone https://github.com/EveNet-HEP/Exotic-Higgs-Study.git
cd Exotic-Higgs-Study
git clone --recursive https://github.com/EveNet-HEP/EveNet-Full.git
```
To run the code, we require the following dependencies:
* Python 3.12 or higher
* EveNet-Full
```aiignore
conda create --prefix [path] python=3.12
conda activate [path]
pip3 install -r requirements.txt
```
Set up the environment variables for EveNet-Full, to enable `wandb` logging, please change the `WANDB_API_KEY` in `src.sh` to your own API key from Weights & Biases. 
Then, source the environment variables:
```aiignore
# This should be run whenever you start a new terminal session to ensure the environment variables are set correctly.
source src.sh
```
Pull the Docker image for the EveNet training:
```aiignore
shifterimg -v pull docker:avencast1994/evenet:1.5
```
## Dataset prepare [Hugging Face]
The dataset used in this study is available on Hugging Face. You can access it using the following link:
[Exotic Higgs Dataset on Hugging Face](https://huggingface.co/datasets/Avencast/EveNet-ExoticHiggs-H2a4b)
(Note: The dataset is large, so ensure you have sufficient storage space before downloading. `--local-dir` option allows you to specify the directory where the dataset will be stored.)
#### 🔹 Download full dataset
```aiignore
hf download Avencast/EveNet-ExoticHiggs-H2a4b \
  --repo-type dataset \
  --local-dir database
```

#### 🔹 Download a subset of the dataset (e.g., certain mass points)
```aiignore
hf download Avencast/EveNet-ExoticHiggs-H2a4b \
  --include "evenet-train/evenet-ma30/**" \
  --include "evenet-test/evenet-ma30/**" \
  --repo-type dataset \
  --local-dir database
```




