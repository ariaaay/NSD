import nibabel as nib
# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')


# img = nib.load("../data/betas_session01.nii.gz")
# data = img.get_data()
# print(data.shape)
# plt.plot(data[50,50,50,:])
# plt.show()

import cortex
import nibabel as nib
import numpy as np
import os
import pickle
import csv
import json

# from util.util import *

out_dir = "/media/tarrlab/NSDdata"
freesurfer_path = "/home/tarrlab/NSD_data/freesurfer"


db = cortex.database.default_filestore
def initiate_subject(subj):
    if "subj0{}".format(subj) not in cortex.db.subjects:
        print(
            "Subjects {} data does not exist in pycortex database. Initiating..".format(
                subj
            )
        )
        # initiate subjects
        cortex.freesurfer.import_subj("subj0" + str(subj), freesurfer_subject_dir=freesurfer_path)
        cortex.freesurfer.import_flat("subj0" + str(subj), "full", freesurfer_subject_dir=freesurfer_path)

def align(subj):
    transform_name = "full"
    transform_path = "{}/subj0{}/transforms/{}".format(db, subj, transform_name)
    if not os.path.isdir(transform_path):  # no transform generated yet
        print("No transform found. Auto aligning...")

        # load a reference slice for alignment'/data2/tarrlab/common/datasets/pycortex_db/sub-CSI{}/func_examples/
        slice_dir = "/home/tarrlab/NSD_data/reference_volumes/subj0{}/".format(subj)
        slice_path = slice_dir + "meanFIRST5.nii.gz"
        # run automatic alignment
        cortex.align.automatic("subj0" + str(subj), transform_name, slice_path)
        # creates a reference transform matrix for this functional run in filestore/db/<subject>/transforms

def main():
    for i in range(8):
        initiate_subject(i+1)
        align(i+1)
        # mask = cortex.utils.get_cortical_mask("subj0{}".format(subj), "full")

main()
