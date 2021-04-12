import numpy as np

import torch
from torch.utils.data import Dataset
from util.model_config import roi_name_dict

# class NSD(Dataset):
#     def __init__(
#         self,
#         roi,
#         idx,
#         transform=transforms.Compose(
#             [
#                 transforms.Resize((64)),
#                 transforms.ToTensor(),
#                 transforms.Normalize(
#                     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
#                 ),
#             ]
#         ),
#     ):

#         from torchvision.datasets import ImageFolder

#         self.image = ImageFolder(
#             "/home/lrg1213/DATA1/NSD_Data/nsddata_stimuli/stimuli/subj1"
#         )
#         self.fmri = np.load("")
#         self.idx = idx
#         self.transform = transform

#     def __getitem__(self, index):
#         img = self.image[self.idx[index]][0]

#         if self.transform is not None:
#             img = self.transform(img)

#         return img, self.fmri[self.idx[index], :]

#     def __len__(self):
#         return self.idx.shape[0]


class NSDBrainOnlyDataset(Dataset):
    def __init__(
        self,
        output_dir,
        subj,
        idx,
        roi,
        roi_num,
    ):   
        self.subj = subj
        self.idx = idx
        self.roi = roi
        self.roi_num = roi_num
        self.output_dir = output_dir
        self.fmri = np.load("%s/cortical_voxels/averaged_cortical_responses_zscored_by_run_subj%02d.npy"% (self.output_dir, self.subj))
        if self.roi is not None:
            self.subset_brain_data()

    def __getitem__(self, index):
        return self.fmri[self.idx[index], :]

    def __len__(self):
        return self.idx.shape[0]

    def subset_brain_data(self):
        roi_mask = np.load(
            "%s/voxels_masks/subj%01d/roi_1d_mask_subj%02d_%s.npy"
            % (self.output_dir, self.subj, self.subj, self.roi)
        )
        roi_dict = roi_name_dict[self.roi]
        mask = roi_mask == self.roi_num
        self.fmri = self.fmri[:, mask]
        #TODO: normalize data?