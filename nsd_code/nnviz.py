"""nnviz.py: optimize CNN objective wrt input image."""
# NOTE: CUDA is _required_ (for pytorch_fft)


import datetime
import pickle
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import torchvision.models as models
import torchvision.transforms as transforms
import pytorch_fft.fft.autograd as fft
from tensorboardX import SummaryWriter



# Channel to optimize (valid values depend on model)
# OPT_CHANNEL = 8

# Make the input larger than required (224x224) to allow cropping
# If all layers of the base model are not required, smaller sizes
#   can also be used

#load labels

BASE_IMSIZE = 224
IMPAD = 16  # Default 16
INPUT_IMSIZE = BASE_IMSIZE + IMPAD
# NUM_CAT = len(affordance_inuse)
# NUM_VOXEL = ?

# Optimization parameters
ITERS = 20000  # Default 10000
STEP_SIZE = 1000  # Default 1000


# ImageNet statistics to normalize images
IMGNET_MEAN = autograd.Variable(torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda())
IMGNET_STD = autograd.Variable(torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda())

# File to save optimized image
OUTPUT_DIR = "vis"

#load trained model
model_path = '/home/yuanw3/AffNet/code/NetDissect/zoo/resnet18_affordance_init_0212.pt'
saved_model = torch.load(model_path)


# Target model construction
base_model = models.vgg16(pretrained=False).cuda()
base_model.fc = nn.Linear(512, NUM_VOXEL)

sd = saved_model['state_dict']
sd_dict = {k.replace('module.',''): v for k, v in sd.items()}
base_model.load_state_dict(sd_dict)
# base_children = list(base_model.children())
# model_part1 = base_children[:6]
# model_part2 = list(base_children[6][0].children())[:1]
# model = nn.Sequential(*(model_part1 + model_part2)).eval()
model = base_model.cuda().eval()

# Input variables: the image is represented as a Fourier transform
#   with real and imaginary components

learning_rates = [0.01, 0.1, 0.5, 1]
gammas = [0.1, 0.3, 0.5, 0.7, 1]
for LR in learning_rates:
    print("Learning rate is %f" % LR)
    for LR_GAMMA in gammas: 
        print("Learning gamma is %f" % LR_GAMMA)
        writer = SummaryWriter("viz_runs/%f_%f" % (LR, LR_GAMMA))

        # for i in tqdm.trange(NUM_CAT):
        for i in range(NUM_CAT):
            OPT_CHANNEL = i
            OUTPUT_FILE = OUTPUT_DIR + "/" + str(datetime.date.today())+ ("_%i_%s_%f_%f.jpg" % (i, affordance_inuse[i],LR, LR_GAMMA))


            xr = autograd.Variable(torch.randn(1, 3, INPUT_IMSIZE, INPUT_IMSIZE).cuda(), requires_grad=True)
            xi = autograd.Variable(torch.randn(1, 3, INPUT_IMSIZE, INPUT_IMSIZE).cuda(), requires_grad=True)

            # Optimizer and learning rate schedule
            opt = optim.Adam([xr, xi], lr=LR)
            lr_sched = optim.lr_scheduler.StepLR(opt, step_size=STEP_SIZE, gamma=LR_GAMMA)

            # Inverse Fourier transform
            invf = fft.Ifft2d()

            # Progress bar
            # pbar = tqdm.trange(ITERS, desc="Optimizing", ncols=80)
            print("optimizing %s" % affordance_inuse[i])
            # Main optimization loop
            # for k in pbar:
            for k in range(ITERS):
                x, _ = invf(xr, xi)  # Transform to image space
                x = (x - x.min()) / (x.max() - x.min())  # Scale values to [0, 1]

                # Take random crop of the image
                cr = torch.LongTensor(2).random_(0, IMPAD+1)
                x = x[:, :, cr[0]:cr[0]+BASE_IMSIZE, cr[1]:cr[1]+BASE_IMSIZE]

                # Get output
                xt = (x - IMGNET_MEAN) / IMGNET_STD
                xt = F.dropout(xt)
                y = model(xt)

                # Loss
                y = y[0, OPT_CHANNEL]
                loss = -y.mean()  # This maximizies the channel; flip sign to minimize instead
                writer.add_scalar(("loss/ %s"%affordance_inuse[i]), loss, k)

                # Optimization step
                lr_sched.step()
                opt.zero_grad()
                loss.backward()
                opt.step()

                # Normalize the Fourier transforms
                ## Mean norm
                xnorm = (((xr**2)+(xi**2)).sum(dim=-1).sum(dim=-1)/(xr.shape[2]*xr.shape[3])).sqrt()
                xnorm = xnorm.unsqueeze(-1).unsqueeze(-1).data
                xr.data /= xnorm
                xi.data /= xnorm

                # Update progress bar
                # pbar.set_description("Optimizing (loss={:.3g})".format(float(loss)))
            # pbar.close()

            # Save final result
            x, _ = invf(xr, xi)
            x = (x - x.min()) / (x.max() - x.min())
            img = transforms.ToPILImage()(x.data.cpu()[0])
            img.save(OUTPUT_FILE)

        writer.close()


