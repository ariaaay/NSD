import argparse

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.shape_base import _take_along_axis_dispatcher
import pandas as pd

from cka import cca_core
from cka.CKA import kernel_CKA, linear_CKA
from util.model_config import *


def make_sym_matrix(X):
    X = X + X.T - np.diag(np.diag(X))
    return X


def pairwise_cka(reps):
    lcka = np.zeros((len(reps), len(reps)))
    kcka = lcka.copy()
    for i, rep1 in enumerate(reps):
        for k, rep2 in enumerate(reps[i:]):
            lcka[i, k + i] = linear_CKA(rep1, rep2)
            kcka[i, k + i] = kernel_CKA(rep1, rep2)

    # to make it symmetric
    lcka = make_sym_matrix(lcka)
    kcka = make_sym_matrix(kcka)
    return lcka, kcka


def cross_cka(reps1, reps2):
    lcka = np.zeros((len(reps1), len(reps2)))
    kcka = lcka.copy()
    for i, rep1 in enumerate(reps1):
        for k, rep2 in enumerate(reps2):
            lcka[i, k] = linear_CKA(rep1, rep2)
            kcka[i, k] = kernel_CKA(rep1, rep2)
    return lcka, kcka


def imshow_cka_results(out, figname, labels, labels2=None):
    n1 = out.shape[0]
    n2 = out.shape[1]

    plt.figure()
    plt.imshow(out)
    plt.yticks(range(0, n1), labels=labels)
    if labels2 is None:
        plt.xticks(range(0, n2), labels=labels)
    else:
        plt.xticks(range(0, n2), labels=labels2, rotation=45)
    plt.colorbar()

    for i in range(n1):
        for j in range(n2):
            text = plt.text(
                j, i, round(out[i, j], 2), ha="center", va="center", color="w"
            )
    plt.savefig(figname)


def run_cka_for_layers(task, layers, layer_labels, subset_idx=None):
    reps = list()
    for layer in layers:
        print("Running CKA for Layers %s..." % layer)
        feature = np.load("%s/%s%s.npy" % (args.feature_dir, task, layer)).squeeze()
        if subset_idx is not None:
            feature = feature[subset_idx, :]
        reps.append(feature)
        lcka, kcka = pairwise_cka(reps)
        np.save("%s/%s_all_layers_linear_cka.npy" % (args.output_dir, task), lcka)
        np.save("%s/%s_all_layers_kernel_cka.npy" % (args.output_dir, task), kcka)

    figname = "%s/%s_linear_cka.png" % (args.figure_dir, task)
    imshow_cka_results(lcka, figname, layer_labels)

    figname = "%s/%s_kernal_cka.png" % (args.figure_dir, task)
    imshow_cka_results(kcka, figname, layer_labels)


def run_cka_across_networks(
    tasks, layers1, layers2, layer_labels1, layer_labels2, subset_idx=None
):
    reps1, reps2 = list(), list()
    for layer1 in layers1:
        feature = np.load(
            "%s/%s%s.npy" % (args.feature_dir, tasks[0], layer1)
        ).squeeze()
        if subset_idx is not None:
            feature = feature[subset_idx, :]
        reps1.append(feature)
    for layer2 in layers2:
        feature = np.load(
            "%s/%s%s.npy" % (args.feature_dir, tasks[1], layer2)
        ).squeeze()
        if subset_idx is not None:
            feature = feature[subset_idx, :]
        reps2.append(feature)

        lcka, kcka = cross_cka(reps1, reps2)
        np.save(
            "%s/%s_vs_%s_linear_cka.npy" % (args.output_dir, tasks[0], tasks[1]), lcka
        )
        np.save(
            "%s/%s_vs_%s_kernel_cka.npy" % (args.output_dir, tasks[0], tasks[1]), kcka
        )

    figname = "%s/%s_vs_%s_linear_cka.png" % (args.figure_dir, tasks[0], tasks[1])
    imshow_cka_results(lcka, figname, layer_labels1, layer_labels2)

    figname = "%s/%s_vs_%s_kernal_cka.png" % (args.figure_dir, tasks[0], tasks[1])
    imshow_cka_results(kcka, figname, layer_labels1, layer_labels2)

def run_cka_across_brain_and_networks(task, layers, brain_data, layer_labels, brain_labels, subset_idx):
    reps = list()
    for layer in layers:
        feature = np.load(
            "%s/%s%s.npy" % (args.feature_dir, task, layer)
        ).squeeze()
        if subset_idx is not None:
            feature = feature[subset_idx, :]
        reps.append(feature)

        lcka, kcka = cross_cka(reps, brain_data)
        np.save(
            "%s/%s_vs_brain_linear_cka.npy" % (args.output_dir, task), lcka
        )
        np.save(
            "%s/%s_vs_brain_kernel_cka.npy" % (args.output_dir, task), kcka
        )
    figname = "%s/%s_vs_brain_linear_cka.png" % (args.figure_dir, task)
    imshow_cka_results(lcka, figname, layers, layer_labels, brain_labels)

    figname = "%s/%s_vs_brain_kernel_cka.png" % (args.figure_dir, task)
    imshow_cka_results(kcka, figname, layers, layer_labels, brain_labels)

def load_roi_mask(roi_name, roi_dict):
    roi_mask = np.load(
        "%s/voxels_masks/subj%01d/roi_1d_mask_subj%02d_%s.npy"
        % (args.output_dir, args.subj, args.subj, roi_name)
    )
    masks, labels = list(), list()
    for i, v in roi_dict.items():        
        if i > 0:
            mask = roi_mask == i
            labels.append(v)
            masks.append(mask)
    return masks, labels


def load_roi_data(roi_names, subset_idx=None):
    brain_data_list = list()
    brain_path = (
        "%s/cortical_voxels/averaged_cortical_responses_zscored_by_run_subj%02d.npy"
        % (args.output_dir, args.subj)
    )

    br_data = np.load(brain_path)
    for roi_name in roi_names:
        roi_label_dict = roi_name_dict[roi_name]
        voxel_masks, labels = load_roi_mask(roi_name, roi_label_dict)

        for mask in voxel_masks:
            if subset_idx is None:
                brain_data_list.append(br_data[:, mask])
            else:
                brain_data_list.append(br_data[subset_idx, mask])

    return brain_data_list, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature_dir",
        type=str,
        default="/user_data/yuanw3/project_outputs/NSD/features/subj1",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/user_data/yuanw3/project_outputs/NSD/output/cka",
    )
    parser.add_argument(
        "--figure_dir", type=str, default="/home/yuanw3/NSD/figures/CKA"
    )
    parser.add_argument("--cka_across_layers", action="store_true", default=False)
    parser.add_argument("--cka_across_networks", type=str, nargs="+", default=None)
    parser.add_argument("--cka_across_layers_and_brain", type=str, default=None)
    parser.add_argument("--subj", default=1, type=int)
    args = parser.parse_args()

    # select the common 1000 images from subject 1
    stim = pd.read_pickle(
        "/lab_data/tarrlab/common/datasets/NSD/nsddata/experiments/nsd/nsd_stim_info_merged.pkl"
    )
    subj1 = stim["subject1"].copy()
    subj1[~stim["shared1000"]] = 0
    subj1_idx = subj1[np.array(stim["subject1"]).astype(bool)]
    subset_idx = np.array(subj1_idx).astype(bool)

    assert len(subset_idx) == 10000
    assert np.sum(subset_idx) == 1000

    # construct task to layer dictionary
    task_layer_dict = dict()
    tasks = ["taskrepr_class_1000", "taskrepr_class_places"]
    basic_layers = [
        "_input_layer1",
        "_input_layer2",
        "_input_layer3",
        "_input_layer5",
        "",
    ]
    for task in tasks:
        task_layer_dict[task] = basic_layers

    tasks = ["taskrepr_edge2d", "taskrepr_edge3d"]
    layers = basic_layers + ["_output_layer1"]
    for task in tasks:
        task_layer_dict[task] = layers

    tasks = ["convnet_alexnet", "place_alexnet"]
    layers = [
        "_conv1_avgpool",
        "_conv2_avgpool",
        "_conv3_avgpool",
        "_conv4_avgpool",
        "_conv5_avgpool",
        "_fc6_avgpool",
        "_fc7_avgpool",
    ]
    for task in tasks:
        task_layer_dict[task] = layers

    # experiment 1 - across all layers for a network
    if args.cka_across_layers:
        for task in task_layer_dict.keys():
            print("Running CKA for task %s..." % task)
            layers = task_layer_dict[task]
            layer_labels = layers.copy()
            layer_labels[4] = "_bottle_neck"
            run_cka_for_layers(task, layers, layer_labels, subset_idx=subset_idx)

    # experiment 2 - across networks
    if args.cka_across_networks is not None:
        tasks = args.cka_across_networks
        print("Running CKA for task %s and %s..." % (tasks[0], tasks[1]))
        layers1 = task_layer_dict[tasks[0]]
        layers2 = task_layer_dict[tasks[1]]
        layer_labels1 = layers1.copy()
        layer_labels1[4] = "_bottle_neck"
        layer_labels2 = layers2.copy()
        layer_labels2[4] = "_bottle_neck"
        run_cka_across_networks(
            tasks, layers1, layers2, layer_labels1, layer_labels2, subset_idx=subset_idx
        )

    if args.cka_across_networks_and_brain:
        tasks = args.cka_across_networks_and_brain
        
        roi_names = ["prf-visualrois", "prf-eccrois", "floc-faces", "floc-places"]
        roi_data, brain_labels = list(), list()
        for roi_name in roi_names:
            data, labels = load_roi_data(roi_names)
            roi_data += data
            brain_labels += labels
        for task in tasks:
            print("Running CKA for task %s and the brain..." % task)
            run_cka_across_brain_and_networks(
                task, layers1, roi_data, layer_labels, brain_labels, subset_idx=subset_idx
            )

