# Exotic-Higgs-Study
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





