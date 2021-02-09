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

    proj_output_dir = "/user_data/yuanw3/project_outputs/NSD/output/rdms"

    image_cat = np.load("../data/NSD_cat_feat.npy")
    image_supercat = np.load("../data/NSD_supcat_feat.npy")

    similarity_features = ["text_embedding_of_labels"]

    if "text_embedding_of_labels" in similarity_features:
        import gensim.downloader as api
        import ssl

        ssl._create_default_https_context = ssl._create_unverified_context
        glove_twitter = api.load("glove-twitter-200")

        # here the rsm is either 80 by 80 or 12 by 12
        emb_cat_sim, _ = make_text_sim_matrix(COCO_cat)
        emb_supercat_sim, super_vocab = make_text_sim_matrix(COCO_super_cat)

        plt.imshow(emb_cat_sim, cmap="YlOrRd")
        plt.colorbar()
        plt.savefig("../Cats/figures/rsm_COCOcat_text_embedding.png")

        plt.imshow(emb_supercat_sim, cmap="YlOrRd")
        plt.colorbar()
        plt.savefig("../Cats/figures/rsm_COCOsupercat_text_embedding.png")

    # Sample images

    # # load random 1000 images idb
    # random_img_ind = np.random.choice(np.arange(image_supercat.shape[0]), 10000)

    # load subj1's 10000 image id
    cocoId_subj1 = np.load("../output/coco_ID_of_repeats_subj01.npy")
    nsd2coco = np.load("../output/NSD2cocoId.npy")
    img_ind = [list(nsd2coco).index(i) for i in cocoId_subj1]
    assert len(img_ind) == 10000

    # sort the images arrays with the order of maximum super cat
    image_supercat_subsample = image_supercat[img_ind, :]
    max_cat = np.argmax(image_supercat_subsample, axis=1)
    max_cat_order = np.argsort(max_cat)
    # plt.hist(max_cat)

    if "object_areas" in similarity_features:
        sorted_image_supercat = image_supercat_subsample[max_cat_order, :]
        sorted_image_supercat_sim_by_image = cosine_similarity(sorted_image_supercat)

        image_cat_subsample = image_cat[img_ind, :]
        sorted_image_cat = image_cat_subsample[np.argsort(max_cat), :]
        sorted_image_cat_sim_by_image = cosine_similarity(sorted_image_cat)

        sorted_image_supercat_sim_by_categories = cosine_similarity(
            sorted_image_supercat.T
        )
        # normalize across categories?

        plt.figure(figsize=(40, 20))
        plt.subplot(1, 2, 1)
        plt.imshow(sorted_image_supercat_sim_by_image, cmap="YlOrRd")
        plt.colorbar()
        plt.title("COCO super categories")
        plt.subplot(1, 2, 2)
        plt.imshow(sorted_image_cat_sim_by_image, cmap="YlOrRd")
        plt.title("COCO basic categories")
        plt.colorbar()
        plt.savefig("../Cats/figures/rsm_COCOsupercat_object_areas.png")

    if "individua_ROIs" in similarity_features:
        PPA = np.load("../output/rdms/subj01_places_PPA.npy")
        OPA = np.load("../output/rdms/subj01_places_OPA.npy")
        RSC = np.load("../output/rdms/subj01_places_RSC.npy")
        FFA1 = np.load("../output/rdms/subj01_faces_FFA-1.npy")
        FFA2 = np.load("../output/rdms/subj01_faces_FFA-1.npy")

        brains = [PPA, OPA, RSC, FFA1, FFA2]

        plt.figure(figsize=(10, 50))
        for i, b in enumerate(brains):
            plt.subplot(1, 5, i + 1)
            plt.imshow(
                b[:, max_cat_order][max_cat_order, :],
                cmap="RdBu_r",
                vmin=-0.5,
                vmax=0.5,
            )

        plt.savefig("../Cats/figures/rsm_COCOsupercat_individual_ROIs.png")

    if "bert_caption" in similarity_features:
        bert = np.load("../features/NSD_bert_all_layer_emb_subj1.npy")
        bert = np.reshape(bert, (10000, 5 * 13 * 768))

        bert_sim = cosine_similarity(bert)
        sorted_bert_sim = bert_sim[max_cat_order, :][:, max_cat_order]

    if "all_rois" in similarity_features:
        all_rois = np.load(
            "../output/rdms/subj01_floc-words_floc-faces_floc-places_prf-visualrois.npy"
        )

    # plot comparison
    plt.figure(figsize=(40, 20))

    plt.subplot(2, 2, 1)
    plt.imshow(sorted_image_supercat_sim_by_image, cmap="YlOrRd")
    plt.title("COCO super categories")
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.imshow(sorted_image_cat_sim_by_image, cmap="YlOrRd")
    plt.title("COCO basic categories")
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.imshow(sorted_bert_sim, "YlOrRd")
    plt.title("BERT features of captions")
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.imshow(
        all_rois[max_cat_order, :][:, max_cat_order],
        cmap="RdBu_r",
        vmin=-0.3,
        vmax=0.3,
    )
    plt.title("All ROIs")
    plt.colorbar()

    plt.savefig("../Cats/figures/rsm/comparison.png")

    # plot imagenet
    plt.figure(figsize=(60, 20))
    layers = [["conv"] * 5 + ["fc"] * 2]
    for i in range(7):

        plt.subplot(2, 4, i + 1)
        imgnet_sim = np.load(
            "%s/subj%02d_convnet_alexnet_%s%01d_avgpool.npy"
            % (proj_output_dir, args.subj, layers[i], i)
        )
        plt.imshow(
            imgnet_sim[max_cat_order, :][:, max_cat_order], cmap="YlOrRd",
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
