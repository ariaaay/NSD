import pickle
import numpy as np
import pandas as pd
import argparse



def extract_first_repeat_img_list(dat, subj, rep=0):
    image_ID_list = list(stim.cocoId[stim.subject1_rep0!=0])
    return image_ID_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subj", type=int)
    parser.add_argument("--rep", type=int, default="0", help="Choose which repeats (0-2)")

    args = parser.parse_args()

    stim = pd.read_pickle("/lab_data/tarrlab/common/datasets/NSD/nsddata/experiments/nsd/nsd_stim_info_merged.pkl")

    image_list = extract_first_repeat_img_list(stim, args.subj, args.rep)
    with open('output/coco_ID_subj%02drep%1d.pkl' % (args.subj, args.rep) , 'wb') as f:
        pickle.dump(image_list, f)