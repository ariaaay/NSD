import pickle
import numpy as np
import pandas as pd
import argparse


def extract_repeat_img_list(stim, subj):
    all_rep_img_list = list()
    for rep in range(3):
        column = 'subject%1d_rep%01d' % (subj, rep)
        image_id_list = list(stim.cocoId[stim[column]!=0])
    all_rep_img_list.append(image_id_list)
    return all_rep_img_list

def extract_repeat_trials_list(stim, subj):
    all_rep_trials_list = list()
    for rep in range(3):
        column = 'subject%1d_rep%01d' % (subj, rep)
        trial_id_list = list(stim[column][stim[column]!=0])
    all_rep_trials_list.append(trial_id_list)
    return all_rep_trials_list

def extract_img_list(stim, subj):
    column = 'subject%1d' % (subj)
    image_id_list = list(stim.cocoId[stim[column]!=0])
    return image_id_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subj", type=int)
    parser.add_argument("--type", type=str)
    parser.add_argument("--all_images", type=bool, help="Return all images a subject see")
    # parser.add_argument("--rep", type=int, default="0", help="Choose which repeats (0-2)")

    args = parser.parse_args()

    stim = pd.read_pickle("/lab_data/tarrlab/common/datasets/NSD/nsddata/experiments/nsd/nsd_stim_info_merged.pkl")

    if args.all_images:
        image_list = extract_img_list(stim, args.subj)
        with open('output/coco_ID_subj%02d.pkl' % (args.subj) , 'wb') as f:
            pickle.dump(image_list, f)

    elif args.type == "cocoId":
        image_list = extract_repeat_img_list(stim, args.subj)
        with open('output/coco_ID_subj%02d.pkl' % args.subj , 'wb') as f:
            pickle.dump(image_list, f)

    elif args.type == "trial":
        trial_list = extract_repeat_trials_list(stim, args.subj)
        with open("output/trials_subj%02d.pkl" % args.subj, 'wb') as f:
            pickle.dump(trial_list, f)

