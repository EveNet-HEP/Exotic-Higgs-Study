import os, sys
from collections import defaultdict

import yaml
import os
import ROOT
import array

import argparse
import torch, h5py
from evenet.control.global_config import global_config
from evenet.network.metrics.assignment import SingleProcessAssignmentMetrics
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback


def process_single_job(pretrain, assignment, segmentation, dataset_size, mass, args, global_config):
    hists = dict()
    ass_results = dict()

    signal_process = f"haa_ma{mass}"
    config_file_path = os.path.join(
        args.farm,
        f'evenet-ma{mass}-{pretrain}-assignment{"-on" if assignment else "-off"}'
        f'-segmentation-{"on" if segmentation else "off"}-dataset_size{dataset_size}.yaml'
    )
    global_config.load_yaml(config_file_path)

    assignment_metrics = SingleProcessAssignmentMetrics(
        device="cpu",
        event_permutations=global_config.event_info.event_permutations[signal_process],
        event_symbolic_group=global_config.event_info.event_symbolic_group[signal_process],
        event_particles=global_config.event_info.event_particles[signal_process],
        product_symbolic_groups=global_config.event_info.product_symbolic_groups[signal_process],
        ptetaphienergy_index=global_config.event_info.ptetaphienergy_index,
        process=signal_process
    )

    if args.network == "evenet":
        file_dir = os.path.join(
            args.store_dir, "predictions",
            f'evenet-ma{mass}-{pretrain}-assignment{"-on" if assignment else "-off"}'
            f'-segmentation-{"on" if segmentation else "off"}-dataset_size{dataset_size}'
        )
        fname = os.path.join(file_dir, "prediction.pt")
        process_id_dict = {-1: "QCD", 0: "haa_maMASS"}

    elif args.network == "spanet":
        file_dir = os.path.join(
            args.store_dir, "predictions",
            f'spanet-ma{mass}-{pretrain}-assignment{"-on" if assignment else "-off"}-dataset_size{dataset_size}'
        )
        fname = os.path.join(file_dir, "predict.h5")
        process_id_dict = {0: "QCD", 1: "haa_maMASS"}

    elif args.network == "spanetv2":
        file_dir = os.path.join(
            args.store_dir, "predictions",
            f'spanetv2-ma{mass}-{pretrain}-assignment{"-on" if assignment else "-off"}-dataset_size{dataset_size}'
        )
        fname = os.path.join(file_dir, "predict.h5")
        process_id_dict = {0: "QCD", 1: "haa_maMASS"}
    else:
        return None  # unsupported network

    if not os.path.exists(fname):
        print(f"\033[91mFile {fname} does not exist, skipping.\033[0m")
        return None

    print(f"\033[92mFile {fname} exists.\033[0m")
    loss = dict()

    # Load predictions
    if args.network == "evenet":
        df = torch.load(fname, map_location='cpu')

        mvascore = torch.concat(
            [torch.nn.functional.softmax(data["classification"]["classification/signal"], dim=1)[..., 1] for data in df],
            dim=0
        )
        process_id = torch.concat([data["subprocess_id"] for data in df], dim=0)
        event_weight = torch.concat([data["event_weight"] for data in df], dim=0)
        for loss_term in df[0]["losses"]:
            loss[loss_term] = torch.concat(
                [
                    data["losses"][loss_term].unsqueeze(0)
                    if data["losses"][loss_term].ndim == 0
                    else data["losses"][loss_term]
                    for data in df
                ],
                dim=0
            )
            loss[loss_term] = loss[loss_term].mean().item()
        loss["total_loss"] = sum(loss[term] for term in loss)
        if assignment and not args.no_assign_evaluate:
            for data in df:
                assignment_target = data['assignment_target'][signal_process]
                assignment_target_mask = data['assignment_target_mask'][signal_process]
                assignment_pred = data['assignment_prediction'][signal_process]["best_indices"]
                assignment_prob = data['assignment_prediction'][signal_process]["assignment_probabilities"]
                detection_prob = data['assignment_prediction'][signal_process]["detection_probabilities"]
                assignment_metrics.update(
                    best_indices=assignment_pred,
                    assignment_probabilities=assignment_prob,
                    detection_probabilities=detection_prob,
                    truth_indices=assignment_target,
                    truth_masks=assignment_target_mask,
                    inputs=None,
                    inputs_mask=None,
                    only_for_log=True
                )

            ass_results[signal_process] = assignment_metrics.summary_log()


    elif args.network == "spanet" or args.network == "spanetv2":
        print(f"\033[92mFile {fname} exists.\033[0m")
        with h5py.File(fname, 'r') as f:
            logits = f['SpecialKey.Classifications/EVENT/signal']
            mvascore = torch.tensor(logits[:, 1])
            process_id = torch.tensor(f['SpecialKey.Inputs/SpecialKey.Classifications/signal'][:])
            event_weight = torch.tensor(f['SpecialKey.Inputs/weight'][:])

            assign_targets = []
            assign_targets_mask = []

            detection_probilities = []
            assignment_probilities = []
            assignment_predictions = []

            if assignment and not args.no_assign_evaluate:

                for event_particle in global_config.event_info.event_particles[signal_process]:
                    assign_targets_event_particle = []
                    assign_predictions_event_particle = []
                    for i, product_particle in enumerate(global_config.event_info.product_particles[signal_process][event_particle]):
                        print("get assignment target from", f"SpecialKey.Inputs/{event_particle}/{product_particle}")
                        assign_target_ = torch.tensor(f[f"SpecialKey.Inputs/{event_particle}/{product_particle}"][:])
                        print("get assignment target from", f"SpecialKey.Inputs/{event_particle}/{product_particle}", assign_target_.shape)
                        assign_targets_event_particle.append(assign_target_)
                        assign_pred_ = torch.tensor(f[f"SpecialKey.Targets/{event_particle}/{product_particle}"][:])
                        print("get assignment prediction from", f"SpecialKey.Targets/{event_particle}/{product_particle}", assign_pred_.shape)
                        assign_predictions_event_particle.append(assign_pred_)
                    assign_targets_event_particle = torch.stack(assign_targets_event_particle, dim=-1)
                    assign_targets.append(assign_targets_event_particle)
                    assign_targets_mask.append(torch.all(assign_targets_event_particle > -0.5, dim=-1))

                    assign_pred_event_particle = torch.stack(assign_predictions_event_particle, dim=-1)
                    assignment_predictions.append(assign_pred_event_particle)
                    assignment_probilities.append(torch.tensor(f[f"SpecialKey.Targets/{event_particle}/assignment_probability"][:]))
                    print("get assignment probability from", f"SpecialKey.Targets/{event_particle}/assignment_probability", assignment_probilities[-1].shape)
                    detection_probilities.append(torch.tensor(f[f"SpecialKey.Targets/{event_particle}/detection_probability"][:]))
                    print("get assignment probability from", f"SpecialKey.Targets/{event_particle}/assignment_probability", assignment_probilities[-1].shape)
                assignment_metrics.update(
                    assignment_predictions,
                    assignment_probabilities=assignment_probilities,
                    detection_probabilities=detection_probilities,
                    truth_indices=assign_targets,
                    truth_masks=assign_targets_mask,
                    inputs=None,
                    inputs_mask=None,
                    only_for_log=True
                )

                ass_results[signal_process] = assignment_metrics.summary_log()

    array_list = dict()
    for id, process in process_id_dict.items():
        array_list[process] = {
            "y": mvascore[process_id == id],
            "w": event_weight[process_id == id]
        }

    for process, data in array_list.items():
        y_all = data["y"].detach().cpu().numpy()
        w_all = data["w"].detach().cpu().numpy()

        hist = ROOT.TH1F('h', "weighted", 1000, 0, 1)
        hist.FillN(len(y_all), array.array('d', y_all), array.array('d', w_all))
        hists[f"{process}_MVAscoreMASS_SR".replace("MASS", str(mass))] = hist

    #if not hists:
    #    print(f"\033[93mNo histograms to write for mass {mass}, skipping.\033[0m")
    #    return None
    return hists, ass_results, loss


def prepare_ntuple(args):
    with open(args.config_workflow) as f:
        control = yaml.safe_load(f)
    # train_file = os.path.join(os.path.abspath(os.path.dirname(args.config_workflow)), control['train_yaml'])
    # with open(train_file) as f:
    #     event_info_file = os.path.join(os.path.abspath(os.path.dirname(train_file)), yaml.safe_load(f)['event_info']['default'])


    masses = control['mass_choice']
    pretrain_choice = control['pretrain_choice'] if args.network == 'evenet' else ['scratch']
    assignment_seg_choice = control['assign_seg_choice'] if args.network == 'evenet' else [[True, False], [False, False]]
    dataset_size_choice = control['dataset_size_choice']

    jobs = []
    results = defaultdict(dict)
    futures = {}
    with ProcessPoolExecutor() as executor:
        for pretrain in pretrain_choice:
            for assignment, segmentation in assignment_seg_choice:
                for dataset_size in dataset_size_choice:
                    if "spanet" in args.network:
                        tag = (
                            f'{args.network}-{pretrain}-assignment{"-on" if assignment else "-off"}'
                            f'-dataset_size{dataset_size}'
                        )

                    else:
                        tag = (
                            f'{args.network}-{pretrain}-assignment{"-on" if assignment else "-off"}'
                            f'-segmentation-{"on" if segmentation else "off"}-dataset_size{dataset_size}'
                        )
                    for mass in masses:
                        future= (
                            executor.submit(
                                process_single_job, pretrain, assignment, segmentation, dataset_size, mass, args, global_config
                            )
                        )
                        futures[future] = (tag, mass)

        for future in as_completed(futures):
            tag, mass = futures[future]
            try:
                result = future.result()
                if tag not in results:
                    results[tag] = dict()
                results[tag][mass] = result
                if result:
                    print(f"✅ Finished writing {tag}[{mass}] → {result}")
            except Exception as e:
                print(f"❌ Job failed for {tag}[{mass}]: {e}")
                traceback.print_exc()

    for tag in results:
        hists = dict()
        ass_results = dict()
        losses = dict()
        for mass in results[tag]:
            if results[tag][mass] is None:
                continue
            hist_, assignment_result_, loss_ = results[tag][mass]
            hists.update(hist_)
            ass_results.update(assignment_result_)
            losses[mass] = loss_

        store_directory = os.path.join(
            args.store_dir, "ntuple", tag
        )
        store_root_name = os.path.join(store_directory, "ntuple.root")
        os.makedirs(store_directory, exist_ok=True)

        f_out = ROOT.TFile(store_root_name, "RECREATE")
        for hist_name, hist in hists.items():
            hist = hist.Clone()
            hist.SetName(hist_name)
            hist.Write()
        f_out.Close()

        store_directory = os.path.join(
            args.store_dir, "loss_summary", tag
        )
        os.makedirs(store_directory, exist_ok=True)
        with open(os.path.join(store_directory, "loss.json"), "w") as f:
            json.dump(losses, f, indent=4)

        if "assignment-on" in tag and not args.no_assign_evaluate:
            ass_store_directory = os.path.join(
                args.store_dir, "assignment_metrics", tag
            )
            os.makedirs(ass_store_directory, exist_ok=True)
            with open(os.path.join(ass_store_directory, "summary.json"), "w") as f:
                json.dump(ass_results, f, indent=4)



def main():
    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("config_workflow", type=str, default="config_workflow.yaml", help="Path to the workflow configuration file")
    parser.add_argument("--farm", type=str)
    parser.add_argument("--store_dir", type=str, default="store", help="Directory to store the output files")
    parser.add_argument("--network", type=str, default="evenet", help="Network name to use for predictions")
    parser.add_argument("--no_assign_evaluate", action="store_true")
    # Parse command-line arguments
    args = parser.parse_args()

    prepare_ntuple(args)

if __name__ == "__main__":
    main()
