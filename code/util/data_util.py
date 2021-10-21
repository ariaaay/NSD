import pickle
import numpy as np
import pandas as pd
from util.model_config import COCO_cat, COCO_super_cat


def load_model_performance(model, task=None, output_root=".", subj=1, measure="corr"):
    if measure == "pvalue":
        measure = "corr"
        pvalue = True
    else:
        pvalue = False

    if task is None:
        output = np.load(
            "%s/output/encoding_results/subj%d/%s_%s_whole_brain.p"
            % (output_root, subj, measure, model),
            allow_pickle=False,
        )
    else:
        output = np.load(
            "%s/output/encoding_results/subj%d/%s_%s_%s_whole_brain.p"
            % (output_root, subj, measure, model, task),
            allow_pickle=False,
        )
    if measure == "corr":
        output = np.array(output)[:, 0]
        if pvalue:
            output = np.array(output)[:, 1]

    return np.array(output)


def load_top1_objects_in_COCO(cid):
    stim = pd.read_pickle(
        "/lab_data/tarrlab/common/datasets/NSD/nsddata/experiments/nsd/nsd_stim_info_merged.pkl"
    )
    cat = np.load("/lab_data/tarrlab/common/datasets/features/NSD/COCO_Cat/cat.npy")

    # extract the nsd ID corresponding to the coco ID in the stimulus list
    stim_ind = stim["nsdId"][stim["cocoId"] == cid]
    # extract the respective features for that nsd ID
    catID_of_trial = cat[stim_ind, :]
    catnm = COCO_cat[np.argmax(catID_of_trial)]
    return catnm


def load_objects_in_COCO(cid):
    stim = pd.read_pickle(
        "/lab_data/tarrlab/common/datasets/NSD/nsddata/experiments/nsd/nsd_stim_info_merged.pkl"
    )
    cat = np.load("/lab_data/tarrlab/common/datasets/features/NSD/COCO_Cat/cat.npy")
    supcat = np.load(
        "/lab_data/tarrlab/common/datasets/features/NSD/COCO_Cat/supcat.npy"
    )

    # extract the nsd ID corresponding to the coco ID in the stimulus list
    stim_ind = stim["nsdId"][stim["cocoId"] == cid]
    # extract the repective features for that nsd ID
    catID_of_trial = cat[stim_ind, :].squeeze()
    supcatID_of_trial = supcat[stim_ind, :].squeeze()
    catnms = []

    assert len(catID_of_trial) == len(COCO_cat)
    assert len(supcatID_of_trial) == len(COCO_super_cat)

    catnms += list(COCO_cat[catID_of_trial > 0])
    catnms += list(COCO_super_cat[supcatID_of_trial > 0])
    return catnms


def load_subset_trials(coco_id_by_trial, cat):
    """
    Returns a list of idx to apply on the 10,000 trials for each subject. These are not trials ID themselves but
    indexs for trials IDS.
    """
    subset_idx = []
    for i, id in enumerate(coco_id_by_trial):
        catnms = load_objects_in_COCO(id)
        if cat in catnms:
            subset_idx.append(i)
    return subset_idx
