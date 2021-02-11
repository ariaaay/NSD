#!/usr/bin/env python
# coding: utf-8

import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity


def make_text_sim_matrix(vocab):
    embs, vocab_in_use = [], []
    for w in vocab:
        try:
            embs.append(glove_twitter.wv[w])
            vocab_in_use.append(w)
        except KeyError:
            try:
                embs.append(glove_twitter.wv[w.replace(" ", "")])
                vocab_in_use.append(w)
            except KeyError:
                print(w)
    emb_sim = cosine_similarity(np.array(embs))
    return emb_sim, vocab_in_use


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subj", default=1)

    args = parser.parse_args()

    COCO_cat = [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]
    COCO_super_cat = [
        "person",
        "vehicle",
        "outdoor",
        "animal",
        "accessory",
        "sports",
        "kitchen",
        "food",
        "furtniture",
        "electronics",
        "appliance",
        "indoor",
    ]

    nsd_output_dir = "/user_data/yuanw3/project_outputs/NSD/output"
    proj_output_dir = nsd_output_dir + "/rdms"
    features_output_dir = "/user_data/yuanw3/project_outputs/NSD/features"

    image_cat = np.load("data/NSD_cat_feat.npy")
    image_supercat = np.load("data/NSD_supcat_feat.npy")

    # # text_embedding_of_labels
    # import gensim.downloader as api
    # import ssl

    # ssl._create_default_https_context = ssl._create_unverified_context
    # glove_twitter = api.load("glove-twitter-200")

    # # here the rsm is either 80 by 80 or 12 by 12
    # emb_cat_sim, _ = make_text_sim_matrix(COCO_cat)
    # emb_supercat_sim, super_vocab = make_text_sim_matrix(COCO_super_cat)

    # plt.imshow(emb_cat_sim, cmap="YlOrRd")
    # plt.colorbar()
    # plt.savefig("../Cats/figures/rsm_COCOcat_text_embedding.png")

    # plt.imshow(emb_supercat_sim, cmap="YlOrRd")
    # plt.colorbar()
    # plt.savefig("../Cats/figures/rsm_COCOsupercat_text_embedding.png")

    # Sample images

    # # load random 1000 images idb
    # random_img_ind = np.random.choice(np.arange(image_supercat.shape[0]), 10000)

    # load subj's 10000 image id
    cocoId_subj = np.load(
        "%s/coco_ID_of_repeats_subj%02d.npy" % (nsd_output_dir, args.subj)
    )
    nsd2coco = np.load("%s/NSD2cocoId.npy" % nsd_output_dir)
    img_ind = [list(nsd2coco).index(i) for i in cocoId_subj]
    assert len(img_ind) == 10000

    # sort the images arrays with the order of maximum super cat
    image_supercat_subsample = image_supercat[img_ind, :]
    max_cat = np.argmax(image_supercat_subsample, axis=1)
    max_cat_order = np.argsort(max_cat)
    # plt.hist(max_cat)

    # object_areas
    sorted_image_supercat = image_supercat_subsample[max_cat_order, :]
    sorted_image_supercat_sim_by_image = cosine_similarity(sorted_image_supercat)

    image_cat_subsample = image_cat[img_ind, :]
    sorted_image_cat = image_cat_subsample[np.argsort(max_cat), :]
    sorted_image_cat_sim_by_image = cosine_similarity(sorted_image_cat)

    sorted_image_supercat_sim_by_categories = cosine_similarity(sorted_image_supercat.T)
    # normalize across categories?

    # plt.figure(figsize=(40, 20))
    # plt.subplot(1, 2, 1)
    # plt.imshow(sorted_image_supercat_sim_by_image, cmap="YlOrRd")
    # plt.colorbar()
    # plt.title("COCO super categories")
    # plt.subplot(1, 2, 2)
    # plt.imshow(sorted_image_cat_sim_by_image, cmap="YlOrRd")
    # plt.title("COCO basic categories")
    # plt.colorbar()
    # plt.savefig("../Cats/figures/rsm_COCOsupercat_object_areas.png")

    # individua_ROIs
    # PPA = np.load("%s/subj%02d_places_PPA.npy" % (proj_output_dir, args.subj))
    # OPA = np.load("%s/subj%02d_places_OPA.npy" % (proj_output_dir, args.subj))
    # RSC = np.load("%s/subj%02d_places_RSC.npy" % (proj_output_dir, args.subj))
    # FFA1 = np.load("%s/subj%02d_faces_FFA-1.npy" % (proj_output_dir, args.subj))
    # FFA2 = np.load("%s/subj%02d_faces_FFA-1.npy" % (proj_output_dir, args.subj))

    # brains = [PPA, OPA, RSC, FFA1, FFA2]

    # plt.figure(figsize=(50, 10))
    # for i, b in enumerate(brains):
    #     plt.subplot(1, 5, i + 1)
    #     plt.imshow(
    #         b[:, max_cat_order][max_cat_order, :], cmap="RdBu_r", vmin=-0.5, vmax=0.5,
    #     )

    # plt.savefig("../Cats/figures/rsm_COCOsupercat_individual_ROIs.png")

    # # bert_caption
    # from util.util import zscore

    # bert = np.load(
    #     "/lab_data/tarrlab/common/datasets/features/NSD/BERT/NSD_bert_all_layer_emb_subj%01d.npy"
    #     % (args.subj)
    # )

    # # layers
    # bert_layer_sim = []
    # for i in range(bert.shape[2]):
    #     blayer = np.reshape(bert[:, :, i, :].squeeze(), (bert.shape[0], bert.shape[1] * bert.shape[3]))
    #     blayer = zscore(blayer, axis=1)
    #     bsim = cosine_similarity(blayer)
    #     bert_layer_sim.append(bsim[max_cat_order,:][:, max_cat_order])

    # plt.figure(figsize=(50, 10))
    # for i in range(13):
    #     plt.subplot(2, 7, i+1)
    #     plt.imshow(bert_layer_sim[i])
    #     plt.title("Layer " + str(i+1))
    #     plt.colorbar()
    # plt.savefig("../Cats/figures/rsm/rsm_bert_by_layers.png")

    # bert = np.reshape(bert, (bert.shape[0], bert.shape[1] * bert.shape[2] * bert.shape[3]))
    # bert = zscore(bert, axis=1)

    # bert_sim = cosine_similarity(bert)
    # sorted_bert_sim = bert_sim[max_cat_order, :][:, max_cat_order]

    # # all_rois
    # all_rois = np.load(
    #     "%s/subj%02d_floc-words_floc-faces_floc-places_prf-visualrois.npy"
    #     % (proj_output_dir, args.subj)
    # )

    # # plot comparison
    # plt.figure(figsize=(40, 20))

    # plt.subplot(2, 2, 1)
    # plt.imshow(sorted_image_supercat_sim_by_image, cmap="YlOrRd")
    # plt.title("COCO super categories")
    # plt.colorbar()

    # plt.subplot(2, 2, 2)
    # plt.imshow(sorted_image_cat_sim_by_image, cmap="YlOrRd")
    # plt.title("COCO basic categories")
    # plt.colorbar()

    # plt.subplot(2, 2, 3)
    # plt.imshow(sorted_bert_sim, "YlOrRd")
    # plt.title("BERT features of captions")
    # plt.colorbar()

    # plt.subplot(2, 2, 4)
    # plt.imshow(
    #     all_rois[max_cat_order, :][:, max_cat_order],
    #     cmap="RdBu_r",
    #     vmin=-0.3,
    #     vmax=0.3,
    # )
    # plt.title("All ROIs")
    # plt.colorbar()

    # plt.savefig("../Cats/figures/rsm/comparison.png")

    # plot imagenet
    plt.figure(figsize=(60, 20))
    layers = ["conv"] * 5 + ["fc"] * 2
    for i in range(7):

        plt.subplot(2, 4, i + 1)
        imgnet = np.load(
            "%s/subj%01d/convnet_alexnet_%s%01d_avgpool.npy"
            % (features_output_dir, args.subj, layers[i], i+1)
        ).squeeze()
        print(imgnet.shape)
        sim = cosine_similarity(imgnet, imgnet)
        
        plt.imshow(
            sim[max_cat_order, :][:, max_cat_order],
            cmap="YlOrRd",
        )
        plt.title("Layer " + str(i + 1))
        plt.colorbar()
    plt.savefig("../Cats/figures/rsm/convnet.png")

    labels = [COCO_super_cat[c] for c in max_cat[max_cat_order]]
    counter = Counter(labels)
    i0 = 0
    for k in counter:
        print("Category: %s [%d - %d]" % (k, i0, counter[k] + i0))
        i0 += counter[k]
