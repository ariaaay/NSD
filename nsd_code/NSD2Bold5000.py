import pickle
import numpy as np
import pandas as pd


stim_info = pd.read_pickle("/lab_data/tarrlab/common/datasets/NSD/nsddata/experiments/nsd/nsd_stim_info_merged.pkl")
NSD_subset = stim_info[stim_info['BOLD5000']==True]
overlap = pd.DataFrame({'cocoId': NSD_subset['cocoId'],
                        'nsdId': NSD_subset['nsdId']})

bold_list = pickle.load(open('/home/yuanw3/ObjectEmbeddingSpace/outputs/convnet_features/convnet_image_orders_fc6.p','rb'))
bold_name_list = [r.split("/")[-1] for r in bold_list]
bold_overlap_list = [i for i, name in enumerate(bold_name_list) if 'COCO' in name and sum(overlap['cocoId'].isin([name.split("_")[-1].split(".")[0][-6:]]))>0]

assert len(bold_overlap_list) == len(overlap['cocoId'])
overlap['BOLD5000Id'] = bold_overlap_list

overlap.to_pickle("/home/yuanw3/NSD/output/NSD2BOLD5000_index.pkl")
