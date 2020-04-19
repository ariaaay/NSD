"""nnviz.py: optimize CNN objective wrt input image."""
# NOTE: CUDA is _required_ (for pytorch_fft)


import datetime
import pickle
import tqdm
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter


# Channel to optimize (valid values depend on model)
# OPT_CHANNEL = 8

# Make the input larger than required (224x224) to allow cropping
# If all layers of the base model are not required, smaller sizes
#   can also be used

# load labels
BASE_IMSIZE = 224
IMPAD = 16  # Default 16
INPUT_IMSIZE = BASE_IMSIZE + IMPAD


# Optimization parameters
ITERS = 20000  # Default 10000
STEP_SIZE = 1000  # Default 1000

parser = argparse.ArgumentParser()
parser.add_argument("--subj", default=1)
parser.add_argument("--model", default="convnet_res50")
parser.add_argument("--num_voxel", default=100)
parser.add_argument("--best_voxels", action="store_true")
parser.add_argument("--roi", type=str, default="")

args = parser.parse_args()

if args.best_voxels:
    try:
        assert args.roi == ""
    except:
        AssertionError("You can only choose one between best voxels or roi masks.")

SUBJ = args.subj
MODEL = args.model

# ImageNet statistics to normalize images
# TODO: change it to COCO size
IMGNET_MEAN = autograd.Variable(
    torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
)
IMGNET_STD = autograd.Variable(
    torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
)

# File to save optimized image
OUTPUT_DIR = "output/optim/subj%d" % SUBJ

# Load weights and voxel index
# number of Face voxels: 699192; words: 2068; visual: 10971

weights = np.load(
    "output/encoding_results/subj%d/weights_%s_whole_brain.npy" % (SUBJ, MODEL)
)

if args.best_voxels: # choose best voxels predicted by the model
    voxel_inds = np.load(
        "output/voxels_masks/subj%d/best_%d_voxel_inds_%s.npy"
        % (SUBJ, args.num_voxel, MODEL)
    )

    NUM_VOXEL = args.num_voxel

else: # choose roi voxels
    mask = np.load(
        "output/voxels_masks/subj%d/roi_1d_mask_subj%02d_%s.npy"
        % (SUBJ, SUBJ, args.roi)
    )
    voxel_inds = np.argwhere(mask>0)
    NUM_VOXEL = len(voxel_inds)

print("Optimizing %d voxels in total..." % NUM_VOXEL)
fc_weight = weights[:, voxel_inds].T

# Target model construction
base_model = models.resnet50(pretrained=True).cuda()
base_model.fc = nn.Linear(2048, NUM_VOXEL)

base_model.fc.weight.data = torch.from_numpy(fc_weight)

model = base_model.cuda().eval()

# Input variables: the image is represented as a Fourier transform
#   with real and imaginary components

learning_rates = [0.01, 0.1, 0.5, 1]
gammas = [0.1, 0.3, 0.5, 0.7, 1]
for LR in learning_rates:
    print("Learning rate is %f" % LR)
    for LR_GAMMA in gammas:
        print("Learning gamma is %f" % LR_GAMMA)
        writer = SummaryWriter("tb_logs/%f_%f" % (LR, LR_GAMMA))

        # for i in tqdm.trange(NUM_CAT):
        for i in range(NUM_VOXEL):
            OPT_CHANNEL = i
            OUTPUT_FILE = (
                OUTPUT_DIR
                + "/"
                + str(datetime.date.today())
                + ("%svoxel_#%s_%f_%f.jpg" % (args.roi+"_", voxel_inds[i], LR, LR_GAMMA))
            )

            xf = autograd.Variable(
                torch.randn(1, 3, INPUT_IMSIZE, 1 + INPUT_IMSIZE // 2, 2).cuda(),
                requires_grad=True,
            )

            # xr = autograd.Variable(torch.randn(1, 3, INPUT_IMSIZE, INPUT_IMSIZE).cuda(), requires_grad=True)
            # xi = autograd.Variable(torch.randn(1, 3, INPUT_IMSIZE, INPUT_IMSIZE).cuda(), requires_grad=True)

            # Optimizer and learning rate schedule
            # opt = optim.Adam([xr, xi], lr=LR)
            opt = optim.Adam([xf], lr=LR)

            lr_sched = optim.lr_scheduler.StepLR(
                opt, step_size=STEP_SIZE, gamma=LR_GAMMA
            )

            # Inverse Fourier transform
            # invf = fft.Ifft2d()

            # Progress bar
            # pbar = tqdm.trange(ITERS, desc="Optimizing", ncols=80)
            print("optimizing %d/100" % i)
            # Main optimization loop
            # for k in pbar:
            for k in range(ITERS):

                # x, _ = invf(xr, xi)  # Transform to image space
                x = torch.irfft(
                    xf,
                    signal_ndim=2,
                    normalized=True,
                    onesided=True,
                    signal_sizes=(INPUT_IMSIZE, INPUT_IMSIZE),
                )

                x = (x - x.min()) / (x.max() - x.min())  # Scale values to [0, 1]

                # Take random crop of the image
                cr = torch.LongTensor(2).random_(0, IMPAD + 1)
                x = x[:, :, cr[0] : cr[0] + BASE_IMSIZE, cr[1] : cr[1] + BASE_IMSIZE]

                # Get output
                xt = (x - IMGNET_MEAN) / IMGNET_STD
                xt = F.dropout(xt)
                y = model(xt)

                # Loss
                y = y[0, OPT_CHANNEL]
                loss = (
                    -y.mean()
                )  # This maximizies the channel; flip sign to minimize instead
                writer.add_scalar(("loss/ voxel #%s" % voxel_inds[i]), loss, k)

                # Optimization step
                opt.zero_grad()
                loss.backward()
                opt.step()
                lr_sched.step()

                # Normalize the Fourier transforms
                ## Mean norm
                # xnorm = (((xr**2)+(xi**2)).sum(dim=-1).sum(dim=-1)/(xr.shape[2]*xr.shape[3])).sqrt()
                # xnorm = xnorm.unsqueeze(-1).unsqueeze(-1).data
                # xr.data /= xnorm
                # xi.data /= xnorm
                xnorm = torch.norm(xf, dim=-1).unsqueeze(-1).data
                # print(xf.data.shape)
                # print(xnorm.shape)
                xf.data /= xnorm

                # Update progress bar
                # pbar.set_description("Optimizing (loss={:.3g})".format(float(loss)))
            # pbar.close()

            # Save final result
            x = torch.irfft(
                xf,
                signal_ndim=2,
                normalized=True,
                onesided=True,
                signal_sizes=(INPUT_IMSIZE, INPUT_IMSIZE),
            )
            x = (x - x.min()) / (x.max() - x.min())
            img = transforms.ToPILImage()(x.data.cpu()[0])
            img.save(OUTPUT_FILE)

        writer.close()
