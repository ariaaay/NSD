import pandas as pd
import h5py
from tqdm import tqdm
from PIL import Image

outpath = "/lab_data/tarrlab/common/datasets/NSD_images"

stim = pd.read_pickle("/lab_data/tarrlab/common/datasets/NSD/nsddata/experiments/nsd/nsd_stim_info_merged.pkl")
name = stim['cocoId']

imgs = h5py.File("/lab_data/tarrlab/common/datasets/NSD/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5")['imgBrick']

for i in tqdm(range(imgs.shape[0])):
    im = Image.fromarray(imgs[i, :, :, :])
    im.save("%s/%s.jpg" % (outpath, stim['cocoId'][i]))