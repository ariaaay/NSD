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

bpath = os.environ["BOLD5000"]
out_dir = "/media/tarrlab/NSD_cortical"

db = cortex.database.default_filestore
def initiate_subject(subj):
    if "subj0{}".format(subj) not in cortex.db.subjects:
        print(
            "Subjects {} data does not exist in pycortex database. Initiating..".format(
                subj
            )
        )
        # initiate subjects
        # if this returns Key Error, manually enter the following lines in ipython works
        cortex.freesurfer.import_subj("subj0" + str(subj), freesurfer_subject_dir=None)

    # transform_name = "full"
    # transform_path = "{}/subj0{}/transforms/{}".format(db, subj, transform_name)
    #
    # if not os.path.isdir(transform_path):  # no transform generated yet
    #     print("No transform found. Auto aligning...")
    #
    #     # load a reference slice for alignment'/data2/tarrlab/common/datasets/pycortex_db/sub-CSI{}/func_examples/
    #     slice_dir = "{}/sub-CSI{}/func_examples/".format(db, subj)
    #     if not os.path.isdir(slice_dir):
    #         os.makedirs(slice_dir)
    #     slice_path = slice_dir + "slice.nii.gz"
    #
    #     try:
    #         nib.load(slice_path)
    #     except FileNotFoundError:
    #         sample_run = (
    #             "{}/derivatives/fmriprep/sub-CSI{}/ses-01/func/sub-CSI{}_ses-01_task-5000scenes_"
    #             "run-01_bold_space-T1w_preproc.nii.gz".format(bpath, subj, subj)
    #         )
    #         img = nib.load(sample_run)
    #         d = img.get_data()
    #         dmean = np.mean(d, axis=3)
    #         sample_slice = nib.Nifti1Image(dmean, img.affine)
    #         nib.save(sample_slice, slice_path)
    #     # run automatic alignment
    #     cortex.align.automatic("sub-CSI" + str(subj), transform_name, slice_path)
    #     # creates a reference transform matrix for this functional run in filestore/db/<subject>/transforms

def main():
    for i in range(8):
        initiate_subject(i)


main()
