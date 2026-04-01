import os, sys
sys.path.append(os.path.abspath("../.."))

import yaml
import os

import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np


def prepare_script(args):

    with open(args.config_workflow) as f:
        control = yaml.safe_load(f)



    config_farm = os.path.abspath(args.farm)
    os.makedirs(config_farm, exist_ok=True)


    working_dir = os.path.abspath(control['working_dir'])
    spanet_dir = os.path.abspath(control['spanet_dir'])
    config_dir = os.path.abspath(os.path.dirname(args.config_workflow))
    config_file = os.path.abspath(args.config_workflow)
    cwd = os.getcwd()

    os.chdir(config_dir)
    with open(control['train_yaml']) as f:
        config_template = yaml.safe_load(f)
    with open(control['predict_yaml']) as f:
        predict_template = yaml.safe_load(f)

    process_json = os.path.abspath(control['process_json'])
    stat_yml = os.path.abspath(control['stat_yml'])

    indir = control['input_dir']

    # ============= Candidates =============

    masses = control['mass_choice']
    pretrain_choice = control['pretrain_choice']
    assignment_seg_choice = control['assign_seg_choice']
    dataset_size_choice = control['dataset_size_choice']

    os.makedirs(os.path.join(args.store_dir, "logs"), exist_ok=True)

    noise_predict_file_list = []
    noise_prepare_command_list = []


    with open(os.path.join(config_farm, "prepare-dataset.sh"), 'w') as f:
        for mass in masses:
            f.write(f"cd {working_dir}\n")
            for split in ["train", "test"]:
                f.write(f"shifter python3 preprocessing/convert_evenet_to_spanet.py {cwd}/configs/event_info_{mass}.yaml --in_dir {args.store_dir}/evenet-{split}/evenet-ma{mass} --store_dir {args.store_dir}/spanet-{split}/spanet-ma{mass}\n")
            for pretrain in pretrain_choice:
                for assignment, segmentation in assignment_seg_choice:
                    for dataset_size in dataset_size_choice:
                        os.chdir(config_dir)
                        config = deepcopy(config_template)
                        config['network']['default'] = os.path.abspath(config['network']['default'])
                        config['event_info']['default'] =  os.path.abspath(config['event_info']['default'].replace("MASS", str(mass)))
                        config['resonance']['default'] = os.path.abspath(config['resonance']['default'])
                        config['options']['default'] = os.path.abspath(pretrain_choice[pretrain]['option'])
                        config['logger']['wandb']['run_name'] = f'evenet-ma{mass}-{pretrain}-assignment{"-on" if assignment else "-off"}-segmentation-{"on" if segmentation else "off"}-dataset_size{dataset_size}'
                        config['options']['Dataset']['normalization_file'] = os.path.join(f"{args.store_dir}/evenet-train/evenet-ma{mass}","normalization.pt")
                        config['options']['Dataset']['dataset_limit'] = dataset_size
                        config['platform']["data_parquet_dir"] =  f"{args.store_dir}/evenet-train/evenet-ma{mass}"
                        if assignment:
                            config["options"]["Training"]["ProgressiveTraining"]["stages"][0]['loss_weights']['assignment'] = [1.0, 1.0]
                            config["options"]["Training"]["Components"]["Assignment"]['include'] = True
                        else:
                            config["options"]["Training"]["ProgressiveTraining"]["stages"][0]['loss_weights']['assignment'] = [0.0, 0.0]
                            config["options"]["Training"]["Components"]["Assignment"]['include'] = False


                        if segmentation:
                            config["options"]["Training"]["ProgressiveTraining"]["stages"][0]['loss_weights']['segmentation'] = [1.0, 1.0]
                            config["options"]["Training"]["Components"]["Segmentation"]['include'] = True
                        else:
                            config["options"]["Training"]["ProgressiveTraining"]["stages"][0]['loss_weights']['segmentation'] = [0.0, 0.0]
                            config["options"]["Training"]["Components"]["Segmentation"]['include'] = False


                        if dataset_size < 0.1:
                            config["options"]["Training"]["epochs"] = 100
                            config["options"]["Training"]["total_epochs"] = 100

                        config["options"]["Training"]["model_checkpoint_save_path"] = os.path.join(args.store_dir, "checkpoints", f'evenet-ma{mass}-{pretrain}-assignment{"-on" if assignment else "-off"}-segmentation-{"on" if segmentation else "off"}-dataset_size{dataset_size}')
                        config["options"]["Training"]["pretrain_model_load_path"] = pretrain_choice[pretrain]['path']

                        predict_config = deepcopy(predict_template)
                        predict_config["platform"]["data_parquet_dir"] = f"{args.store_dir}/evenet-test/evenet-ma{mass}"
                        predict_config["options"]["default"] = os.path.abspath(predict_config["options"]["default"])
                        predict_config["options"]["prediction"]["output_dir"] = os.path.join(args.store_dir, "predictions", f'evenet-ma{mass}-{pretrain}-assignment{"-on" if assignment else "-off"}-segmentation-{"on" if segmentation else "off"}-dataset_size{dataset_size}')

                        if segmentation:
                            predict_config["options"]["Training"]["Components"]["Segmentation"]['include'] = True
                        else:
                            predict_config["options"]["Training"]["Components"]["Segmentation"]['include'] = False

                        if assignment:
                            predict_config["options"]["Training"]["Components"]["Assignment"]['include'] = True
                        else:
                            predict_config["options"]["Training"]["Components"]["Assignment"]['include'] = False



                        ckpt_dir =  os.path.join(args.store_dir, "checkpoints", f'evenet-ma{mass}-{pretrain}-assignment{"-on" if assignment else "-off"}-segmentation-{"on" if segmentation else "off"}-dataset_size{dataset_size}')

                        predict_config["options"]["Training"]["model_checkpoint_load_path"] = ckpt_dir
                        predict_config["options"]["Dataset"]["normalization_file"] = os.path.join(f"{args.store_dir}/evenet-train/evenet-ma{mass}", "normalization.pt")

                        predict_config["network"]["default"] = os.path.abspath(predict_config["network"]["default"])
                        predict_config["event_info"]["default"] = os.path.abspath(predict_config["event_info"]["default"].replace("MASS", str(mass)))
                        predict_config["resonance"]["default"] = os.path.abspath(predict_config["resonance"]["default"])

                        file_path = os.path.join(config_farm, f'evenet-ma{mass}-{pretrain}-assignment{"-on" if assignment else "-off"}-segmentation-{"on" if segmentation else "off"}-dataset_size{dataset_size}.yaml')
                        file_path_predict = os.path.join(config_farm, f'evenet-ma{mass}-{pretrain}-assignment{"-on" if assignment else "-off"}-segmentation-{"on" if segmentation else "off"}-dataset_size{dataset_size}_predict.yaml')
                        os.chdir(cwd)
                        with open(file_path, 'w') as fout:
                            yaml.dump(config, fout)
                        with open(file_path_predict, 'w') as fout:
                            yaml.dump(predict_config, fout)

                        # Noise study addition
                        if args.noise_study_number > 0:
                            random_state = np.random.RandomState(args.seed)
                            for seed_number in range(args.noise_study_number):
                                if (seed_number == 0):
                                    noise_level_variation = 0.0 # always have nominal to be compared
                                else:
                                    noise_level_variation = np.clip(random_state.normal(loc=0.0, scale=args.noise_level), a_min=-0.1, a_max=0.1)
                                for variation in ['up', 'down']:
                                    if variation == 'up':
                                        noise_level_variation = noise_level_variation
                                    else:
                                        noise_level_variation = -noise_level_variation
                                    noisy_file_path_predict = os.path.join(config_farm, f'evenet-ma{mass}-{pretrain}-assignment{"-on" if assignment else "-off"}-segmentation-{"on" if segmentation else "off"}-dataset_size{dataset_size}_predict_noise{noise_level_variation:.6f}_seed{seed_number}-{variation}.yaml')
                                    noisy_predict_config = deepcopy(predict_config)
                                    variation_inputdir = f"{args.store_dir}/evenet-test/evenet-ma{mass}-onefile/"
                                    variation_input = os.path.join(variation_inputdir, f"data_Combined_Balanced_run_0.parquet")
                                    variation_outdir = f"{args.store_dir}/evenet-test/evenet-ma{mass}-onefile/noise_level_variation{noise_level_variation:.6f}_{variation}/"

                                    prepare_command = f"python3 systematic_shift.py --inputdir {variation_inputdir} --outdir {variation_outdir} --shift {noise_level_variation:.6f} \n"
                                    if prepare_command not in noise_prepare_command_list:
                                        noise_prepare_command_list.append(prepare_command)

                                    noisy_predict_config["platform"]["data_parquet_dir"] = variation_outdir
                                    noisy_predict_config["options"]["prediction"]["output_dir"] = os.path.join(
                                        args.store_dir,
                                        "noise-predictions",
                                        f'evenet-ma{mass}-{pretrain}-assignment{"-on" if assignment else "-off"}-segmentation-{"on" if segmentation else "off"}-dataset_size{dataset_size}/predict_noise{noise_level_variation}'
                                    )
                                    # noisy_predict_config['options']['prediction']['limit_predict_batches'] = args.noise_batch
                                    # noisy_predict_config['options']['prediction']['save_intermediate'] = True
                                    noisy_predict_config['platform']['number_of_workers'] = 1
                                    with open(noisy_file_path_predict, 'w') as fout:
                                        yaml.dump(noisy_predict_config, fout)
                                    noise_predict_file_list.append(noisy_file_path_predict)

    with open(os.path.join(config_farm, "prepare-noise-dataset.sh"), 'w') as f:
        for command in noise_prepare_command_list:
            f.write(command)

    with open(os.path.join(config_farm, "train-evenet.sh"), 'w') as f:
        # f.write(f'python3 Split_dataset.py {" ".join(indir)} --output_dir {args.store_dir}\n')
        # f.write(f'python3 Prepare_preprocess_config.py {config_file} --store_dir {args.store_dir} --farm {config_farm}\n')
        for mass in masses:
            for pretrain in pretrain_choice:
                for assignment, segmentation in assignment_seg_choice:
                    for dataset_size in dataset_size_choice:
                        file_path = os.path.join(config_farm, f'evenet-ma{mass}-{pretrain}-assignment{"-on" if assignment else "-off"}-segmentation-{"on" if segmentation else "off"}-dataset_size{dataset_size}.yaml')
                        file_path_predict = os.path.join(config_farm, f'evenet-ma{mass}-{pretrain}-assignment{"-on" if assignment else "-off"}-segmentation-{"on" if segmentation else "off"}-dataset_size{dataset_size}_predict.yaml')
                        os.chdir(cwd)
                        job_name = f'evenet-ma{mass}-{pretrain}-assignment{"-on" if assignment else "-off"}-segmentation-{"on" if segmentation else "off"}-dataset_size{dataset_size}'

                        f.write(f"cd {working_dir} && ")
                        f.write(f"python3 evenet/train.py {file_path} --ray_dir {args.ray_dir} {'--load_all' if dataset_size < 0.2 else ''} \n")
                        # f.write(f"python3 evenet/predict.py {os.path.abspath(file_path_predict)} \n")
    with open(os.path.join(config_farm, "predict-evenet.sh"), 'w') as f:
        # f.write(f'python3 Split_dataset.py {" ".join(indir)} --output_dir {args.store_dir}\n')
        # f.write(f'python3 Prepare_preprocess_config.py {config_file} --store_dir {args.store_dir} --farm {config_farm}\n')
        for mass in masses:
            for pretrain in pretrain_choice:
                for assignment, segmentation in assignment_seg_choice:
                    for dataset_size in dataset_size_choice:
                        file_path = os.path.join(config_farm, f'evenet-ma{mass}-{pretrain}-assignment{"-on" if assignment else "-off"}-segmentation-{"on" if segmentation else "off"}-dataset_size{dataset_size}.yaml')
                        file_path_predict = os.path.join(config_farm, f'evenet-ma{mass}-{pretrain}-assignment{"-on" if assignment else "-off"}-segmentation-{"on" if segmentation else "off"}-dataset_size{dataset_size}_predict.yaml')
                        os.chdir(cwd)
                        job_name = f'evenet-ma{mass}-{pretrain}-assignment{"-on" if assignment else "-off"}-segmentation-{"on" if segmentation else "off"}-dataset_size{dataset_size}'

                        f.write(f"cd {working_dir} && ")
                        f.write(f"python3 evenet/predict.py {os.path.abspath(file_path_predict)} --ray_dir {args.ray_dir} \n")

    with open(os.path.join(config_farm, "prepare-noise-up.sh"), 'w') as f:
        for noise_file in noise_predict_file_list:
            if "up" not in noise_file:
                continue
            f.write(f"cd {working_dir} && ")
            f.write(f"python3 evenet/predict.py {os.path.abspath(noise_file)} --ray_dir {args.ray_dir} \n")
    with open(os.path.join(config_farm, "prepare-noise-down.sh"), 'w') as f:
        for noise_file in noise_predict_file_list:
            if "down" not in noise_file:
                continue
            f.write(f"cd {working_dir} && ")
            f.write(f"python3 evenet/predict.py {os.path.abspath(noise_file)} --ray_dir {args.ray_dir} \n")

    with open(os.path.join(config_farm, "summary.sh"), 'w') as f:
        f.write(f"cd {cwd}\n")
        f.write(f"python3 Produce_ntuple.py {config_file} --store_dir {args.store_dir} --farm {args.farm} \n")
        f.write(f"python3 Produce_ntuple.py {config_file} --store_dir {args.store_dir} --network spanet --farm {args.farm}\n")

        for pretrain in pretrain_choice:
            for assignment, segmentation in assignment_seg_choice:
                for dataset_size in dataset_size_choice:
                    store_directory = os.path.join(args.store_dir, "ntuple",
                                                   f'evenet-{pretrain}-assignment{"-on" if assignment else "-off"}-segmentation-{"on" if segmentation else "off"}-dataset_size{dataset_size}')
                    out_directory = os.path.join(args.store_dir, "fit",
                                                   f'evenet-{pretrain}-assignment{"-on" if assignment else "-off"}-segmentation-{"on" if segmentation else "off"}-dataset_size{dataset_size}')
                    f.write(f"python3 Statistics_test.py --Lumi {args.Lumi} --signal all --process_json {process_json} --sourceFile {store_directory}/ntuple.root --observable MVAscoreMASS --config_yml {stat_yml} --outdir {out_directory} --log_scale & \n")
        f.write(f"python3 Produce_ntuple.py {config_file} --store_dir {args.store_dir} --network spanet\n")
        for assignment, _ in assignment_seg_choice:
            for dataset_size in dataset_size_choice:
                store_directory = os.path.join(args.store_dir, "ntuple",
                                               f'spanetv2-scratch-assignment{"-on" if assignment else "-off"}-dataset_size{dataset_size}')
                out_directory = os.path.join(args.store_dir, "fit",
                                               f'spanetv2-scratch-assignment{"-on" if assignment else "-off"}-dataset_size{dataset_size}')
                f.write(f"python3 Statistics_test.py --Lumi {args.Lumi} --signal all --process_json {process_json} --sourceFile {store_directory}/ntuple.root --observable MVAscoreMASS --config_yml {stat_yml} --outdir {out_directory} --log_scale & \n")
        f.write(f"python3 Summary_Limit.py --store_dir {args.store_dir}\n")

    with open(os.path.join(config_farm, "train_spanet.sh"), 'w') as f:
        for mass in masses:
            f.write(f"cd {spanet_dir}\n")
            for assignment in [True, False]:
                for dataset_size in dataset_size_choice:
                    dataset_dir = f"{args.store_dir}/spanet-train/spanet-ma{mass}"
                    dataset = f"{dataset_dir}/data.h5"
                    run_name = f'spanet-ma{mass}-scratch-assignment{"-on" if assignment else "-off"}-dataset_size{dataset_size}'
                    options_file = "options_files/exotic_higgs_decay/full_training.json" if assignment else "options_files/exotic_higgs_decay/full_training-cls.json"
                    log_dir = os.path.join(args.store_dir)

                    epochs = 100 if dataset_size < 0.1 else 50

                    if dataset_size > 0.09:
                        batch_size = 2048
                    elif dataset_size > 0.02:
                        batch_size = 1024
                    else:
                        batch_size = 512

                    f.write(
                        f"cd {spanet_dir}; python3 -m spanet.train --event_file event_files/haa_ma{mass}.yaml -tf {dataset} --options_file {options_file} --log_dir {log_dir} --run_name {run_name} --epochs {epochs} --gpus 4 --limit_dataset {dataset_size * 100} -b {batch_size} --project {control['spanet']['project']} \n")


    with open(os.path.join(config_farm, "predict_spanet.sh"), 'w') as f:
        for mass in masses:
            f.write(f"cd {spanet_dir}\n")
            for assignment, _ in assignment_seg_choice:
                for dataset_size in dataset_size_choice:
                    dataset_dir = f"{args.store_dir}/spanet-train/spanet-ma{mass}"
                    dataset = f"{dataset_dir}/data.h5"
                    run_name = f'spanetv2-ma{mass}-scratch-assignment{"-on" if assignment else "-off"}-dataset_size{dataset_size}'
                    options_file =  "options_files/exotic_higgs_decay/full_training_default_setting.json" if assignment else "options_files/exotic_higgs_decay/full_training-cls_default_setting.json"
                    log_dir = os.path.join(args.store_dir)
#                    f.write(f"python3 -m spanet.train --event_file event_files/haa_ma{mass}.yaml -tf {dataset} --options_file {options_file} --log_dir {log_dir} --run_name {run_name} --epochs 50 --gpus 4 --limit_dataset {dataset_size * 100} --project {control['spanet']['project']} \n")
                    f.write(f"python3 -m spanet.predict {log_dir}/checkpoints/{run_name} {args.store_dir}/predictions/{run_name}/predict.h5 -tf {args.store_dir}/spanet-test/spanet-ma{mass}/data.h5  --event_file event_files/haa_ma{mass}.yaml --batch_size 1024 --gpu\n")
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("config_workflow", type=str, default="config_workflow.yaml", help="Path to the workflow configuration file")
    parser.add_argument("--store_dir", type=str, default="store", help="Directory to store the output files")
    parser.add_argument("--ray_dir", type=str, default="ray", help="Directory for Ray cluster")
    parser.add_argument("--farm", type=str, default="config_farm", help="Directory to store the configuration files")
    parser.add_argument("--Lumi", type=float, default=1000.0, help="Luminosity for the simulation")
    parser.add_argument("--noise_study_number", type=int, default=0, help="Noise study number")
    parser.add_argument("--noise_level", type=float, default=0.0, help="Noise level for the simulation")
    # parser.add_argument("--noise_batch", type=int, default=10, help="Target file for noise addition")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for noise generation")
    # Parse command-line arguments
    args = parser.parse_args()
    prepare_script(args)

if __name__ == "__main__":
    main()
