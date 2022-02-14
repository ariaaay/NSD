from __future__ import print_function
from __future__ import division

import clip
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, models, transforms
from torch.utils.tensorboard import SummaryWriter

# from skorch.callbacks import TensorBoard
# from sklearn.model_selection import GridSearchCV
# from skorch.helper import SliceDataset

from transformers import AutoTokenizer, BertModel, BertConfig
import matplotlib.pyplot as plt
import time
import os
import copy
import argparse

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

num_classes = 768  # should set to dim of BERT embedding
num_epochs = 400

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = False


def set_parameter_requires_grad(model, feature_extract):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False


def prepare_dataloader(num_workers=4, train_batch_size=200, val_batch_size=256):
    # Just normalization for validation
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        ),
    }

    print("Initializing Datasets and Dataloaders...")

    train_set = datasets.CocoCaptions(
        root="/lab_data/tarrlab/common/datasets/COCO/train2017/",
        annFile="/lab_data/tarrlab/common/datasets/coco_annotations/captions_train2017.json",
        transform=data_transforms["train"],
        target_transform=np.random.choice,
    )
    val_set = datasets.CocoCaptions(
        root="/lab_data/tarrlab/common/datasets/COCO/val2017/",
        annFile="/lab_data/tarrlab/common/datasets/coco_annotations/captions_val2017.json",
        transform=data_transforms["val"],
        target_transform=np.random.choice,
    )

    print("Number of samples: ", len(train_set))
    img, target = train_set[3]  # load 4th sample

    print("Image Size: ", img.size())
    print(target)

    train_sampler = torch.utils.data.RandomSampler(train_set)
    val_sampler = torch.utils.data.SequentialSampler(val_set)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=train_batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=val_batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
    )

    dataloaders_dict = {"train": train_loader, "val": val_loader}
    return dataloaders_dict, train_set, val_set


def calculate_corrects(similarity):
    target = torch.arange(similarity.shape[0]).to(device)
    _, preds = torch.max(similarity, 1)
    return torch.sum(preds == target)


def train_model(
    model,
    lmodel,
    clip_model,
    t,
    tokenizer,
    dataloaders,
    criterion,
    optimizer,
    num_epochs=25,
):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
                clip_running_corrects = 0

            running_loss = 0.0
            running_corrects = 0

            for inputs, captions in dataloaders[phase]:
                inputs = inputs.to(device)
                # print(type(captions))
                encoded = tokenizer(
                    list(captions), padding=True, truncation=True, return_tensors="pt"
                ).to(device)
                # tokens = tokenizer.encode(captions, add_special_tokens=False)
                with torch.no_grad():
                    text_out = lmodel(
                        encoded["input_ids"], token_type_ids=encoded["token_type_ids"]
                    )
                    language_emb = text_out.last_hidden_state[:, -1, :].squeeze()
                    # print(language_emb.size())
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    I_e = nn.functional.normalize(outputs, dim=-1)
                    T_e = nn.functional.normalize(language_emb, dim=-1)
                    similarity = I_e @ T_e.T
                    torch.clamp(t, min=1 / 100)
                    logits = similarity * torch.exp(t)
                    target = torch.arange(I_e.shape[0]).to(device)
                    loss = criterion(logits, target)
                    corrects = calculate_corrects(similarity)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                if phase == "val":
                    # evaluate accuracy with CLIP as well
                    text_tokens = clip.tokenize(captions).to(device)
                    with torch.no_grad():
                        image_features = clip_model.encode_image(inputs).float()
                        text_features = clip_model.encode_text(text_tokens).float()
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        text_features /= text_features.norm(dim=-1, keepdim=True)
                        similarity = text_features @ image_features.T
                        clip_running_corrects += calculate_corrects(similarity)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += corrects

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))
            writer.add_scalar("Loss/{}".format(phase), epoch_loss, epoch)
            writer.add_scalar("Accuracy/{}".format(phase), epoch_acc, epoch)
            writer.add_scalar(
                "Temperature/{}".format(phase), t.data.cpu().numpy(), epoch
            )

            if phase == "val":
                epoch_clip_acc = clip_running_corrects.double() / len(
                    dataloaders[phase].dataset
                )

                print("CLIP Acc: {:.4f}".format(epoch_clip_acc))
                writer.add_scalar(
                    "CLIP Accuracy/{}".format(phase), epoch_clip_acc, epoch
                )

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(
                        best_model_wts,
                        "/user_data/yuanw3/project_outputs/NSD/output/finetune/resnet_finetune.p",
                    )

        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))

    model.load_state_dict(best_model_wts)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchsize", type=int, default=200)
    parser.add_argument("--lr", default=0.005)
    # parser.add_argument("--temp", default=1)

    args = parser.parse_args()

    writer = SummaryWriter(
        log_dir="runs/batchsize_%d_LR_%.4f_learned_temp" % (args.batchsize, args.lr)
    )

    model = models.resnet50(pretrained=True)

    set_parameter_requires_grad(model, feature_extract=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    input_size = 224

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    t = torch.tensor(1 / 0.07)

    params_to_update = list(model.parameters()) + [t]

    # optimizer_ft = optim.SGD(params_to_update, lr=LR, momentum=0.9)
    optimizer_ft = optim.Adam(params_to_update, lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    dataloaders_dict, train_set, val_set = prepare_dataloader(
        train_batch_size=args.batchsize
    )
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    # tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')    # Download vocabulary from S3 and cache.
    # text_pineline = lambda x: tokenizer.encode(x, add_special_tokens=True)
    bert = torch.hub.load(
        "huggingface/pytorch-transformers", "model", "bert-base-cased"
    )  # Download model and configuration from S3 and cache.
    bert = bert.to(device)
    clip_model, clip_preprocess = clip.load("ViT-B/32")
    clip_model = clip_model.to(device)

    # #grid search
    # model.set_params(train_split=False, verbose=0, callbacks=[])
    # params = {
    #     'lr': np.logspace(-5, -3, 3),
    #     # 'batch_size': [10, 20],
    #     'temperature': np.logspace(-3, 0, 4),
    # }
    # gs = GridSearchCV(model, param_grid=params, refit=False, cv=3, scoring='accuracy', verbose=2)
    # train_sliceable = SliceDataset(train_set)
    # y_train = np.array([y for _, y in iter(train_set)])

    # gs.fit(train_sliceable, y_train)
    # print("best score: {:.3f}, best params: {}".format(gs.best_score_, gs.best_params_))

    # best_temp = gs.best_params_["temperature"]
    # best_lr = gs.best_params_["lr"]

    train_model(
        model,
        bert,
        clip_model,
        t,
        tokenizer,
        dataloaders_dict,
        criterion,
        optimizer_ft,
        num_epochs=num_epochs,
    )
