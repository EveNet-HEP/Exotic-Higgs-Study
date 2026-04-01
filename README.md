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

## Download Pre-trained Models
To download the pre-trained models from [Hugging Face Hub](https://huggingface.co/Avencast/EveNet), you can use the following command (`local-dir` could be replaced with any path you want)
```aiignore
hf download Avencast/EveNet --local-dir pretrain-weights
# nominal ckpt: checkpoints.20M.a4.last.ckpt
# SSL ckpt: SSL.20M.last.ckpt
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

## Script Preparation
The training and evaluation scripts are controlled by `config/workflow.yaml`. You can modify the parameters in this file to customize the training and evaluation process.
```yaml
working_dir: [PATH_TO_EveNet_WORKING_DIRECTORY]
spanet_dir: [PATH_TO_SPANet_WORKING_DIRECTORY]
train_yaml: train.yaml [relative path under configs dir] # don't need to change if you use the provided train.yaml
predict_yaml: predict.yaml [relative path under configs dir] # don't need to change if you use the provided predict.yaml
process_json: process.json [relative path under configs dir] # don't need to change if you use the provided process.json
stat_yml: Statistics_Test.yml [relative path under configs dir] # don't need to change if you use the provided Statistics_Test.yml
account: # your project account
email: # your email account
time: 04:00:00 # job time
job_flavor: regular # job flavor
spanet:
  project: ExoticHiggsDecay # wandb project name

pretrain_choice:
  scratch:
    path: null 
    option: options.yaml # use this option file if training from scratch
  pretrain:
    path: /global/cfs/cdirs/m5019/avencast/Checkpoints/EveNet-20M-20250901/last.ckpt
    option: options_pretrain.yaml # use this option file if training from a pretrained model

assign_seg_choice:
  - [true, true] # first bool: assign, second bool: segment
  - [true, false]
  - [false, true]
  - [false, false]

dataset_size_choice:
  - 0.01 # fraction of dataset size to use for training
  - 0.03
  - 0.1
  - 0.3
  - 1.0

mass_choice:
  - 30 # signal mass of a in GeV
  - 40
  - 60
```
Then you can generate the workflow scripts using the following command:
```aiignore
python3 Make_script.py configs/workflow.yaml --store_dir database --ray_dir [tmp dir] --Lumi 300 --farm Farm
```


## Reformatting the data
To reformat the data for baseline `spanet` training, you can use the following command:
```aiignore
sh Farm/prepare-dataset.sh
```
This script typically performs the following steps:
```aiignore
python3 convert_evenet_to_spanet.py [event-info.yaml] --in_dir database --store_dir [target dir]
```
## Train Evenet
All the needed training commands are saved in `Farm/train-evenet.sh`, please note that sequentially running all the commands in this script 
is very time-consuming, better to prepare batch or parallel scripts to run them. 
#### Details
This file basically contains the following command template for training EveNet:
```aiignore
cd /global/u1/t/tihsu/Exotic-Higgs-Study/EveNet-Full; \
  shifter --image=docker:avencast1994/evenet:1.5 python3 scripts/train.py [train yaml] --ray_dir [tmp dir]
```

## Predict EveNet (on Test Dataset)
Similarly, all the needed prediction commands are saved in `Farm/predict-evenet.sh`, 
please note that sequentially running all the commands in this script may also be time-consuming,
better to prepare batch or parallel scripts to run them.
#### Details
This file basically contains the following command template for predicting with EveNet:
```aiignore
cd /global/u1/t/tihsu/Exotic-Higgs-Study/EveNet-Full; \
 shifter --image=docker:avencast1994/evenet:1.5 python3 scripts/predict.py [predict yaml]
```

## Evaluate the results
To evaluate the results, you can use the following command:
```aiignore
sh Farm/summary.sh
```
#### Details
Basically, this script performs the following steps:
```aiignore
# Produce histograms for the signal and background predictions
python3 Produce_ntuple.py /global/u1/t/tihsu/Exotic-Higgs-Study/configs/workflow.yaml --store_dir /pscratch/sd/t/tihsu/test_exotic/ --farm Farm
# For SPANet evaluation, add the network argument to specify the model used for evaluation
python3 Produce_ntuple.py /global/u1/t/tihsu/Exotic-Higgs-Study/configs/workflow.yaml --store_dir /pscratch/sd/t/tihsu/test_exotic/ --farm Farm --network spanet
```
And run the evaluation script to calculate the limits:
```aiignore
python3 Statistics_test.py \
 --Lumi 300 \
 --signal all \
 --process_json configs/process.json \
 --sourceFile [outdir]/ntuple/[tag]/ntuple.root \
 --observable MVAscoreMASS \
 --config_yml configs/Statistics_Test.yml \
 --outdir [outdir]/fit/[tag] --log_scale &
```


## Summarize the results
To summarize the results, you can use the following command:
```aiignore
python3 Summary_Limit.py --store_dir [database dir]]
```