import pickle
import json
from matplotlib import test
import numpy as np
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt


import torch
import clip

from sklearn.model_selection import train_test_split
from util.util import r2_score

from util.coco_utils import load_captions
from featureprep.feature_prep import get_preloaded_features

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def get_t2i_map():

    file_name = "%s/output/clip/text_to_image_ridge_model.pkl" % args.output_root
    try:
        with open(file_name, "rb") as file:
            clf = pickle.load(file)
    except FileNotFoundError:
        all_coco_ids = np.load(
            "%s/output/coco_ID_of_repeats_subj%02d.npy" % (args.output_root, 1)
        )
        image_activations = get_preloaded_features(
            1,
            all_coco_ids,
            "clip",
            layer=None,
            features_dir=args.features_dir,
        ).squeeze()
        text_activations = get_preloaded_features(
            1,
            all_coco_ids,
            "clip_text",
            layer=None,
            features_dir=args.features_dir,
        )
        text_activations = np.array(text_activations).squeeze()
        train_idx, test_idx = train_test_split(
            np.arange(text_activations.shape[0]), test_size=0.15, random_state=42
        )
        # print(text_activations.shape)

        from sklearn.linear_model import Ridge

        clf = Ridge(alpha=1.0)
        clf.fit(text_activations[train_idx], image_activations[train_idx])
        file_name = "%s/output/clip/text_to_image_ridge_model.pkl" % args.output_root
        with open(file_name, "wb") as file:
            pickle.dump(clf, file)

        score = clf.score(text_activations[test_idx], image_activations[test_idx])
        print("text to image score on 500 test images is: " + str(score))
    return clf


def compute_acc(scores):
    prediction_idx = np.argmax(scores, axis=1)
    top5_count, rank_count = 0, 0
    n = scores.shape[0]
    for gi in range(n):
        sample_preds = np.argsort(scores[gi, :])[::-1]
        if gi in sample_preds[:5]:
            top5_count += 1
        pred_rank = np.where(sample_preds == gi)[0]
        rank_count += (
            n - pred_rank
        ) / n  # 1000 - rank in prediction; correct prediction would yield 1

    acc_top1 = np.sum(prediction_idx == np.arange(n)) / n
    acc_top5 = top5_count / n
    acc_rank = rank_count / n

    return [acc_top1, acc_top5, acc_rank], prediction_idx


def decode_captions_for_voxels_t2i(subj):
    gt_subset_n = 1000  # take the first n samples from test set as candidate answers

    all_captions = []
    all_coco_ids = np.load(
        "%s/output/coco_ID_of_repeats_subj%02d.npy" % (args.output_root, subj)
    )

    train_idx, test_idx = train_test_split(
        np.arange(len(all_coco_ids)), test_size=0.15, random_state=42
    )
    test_idx = test_idx[:gt_subset_n]

    test_coco_ids = all_coco_ids[test_idx]
    for cid in tqdm(test_coco_ids):
        captions = load_captions(cid)
        all_captions.append(captions[0])

    mask = np.load(
        "%s/output/pca/clip/best_20000_nc/pca_voxels/pca_voxels_subj%02d.npy"
        % (args.output_root, subj)
    )
    # mask = np.load("%s/output/voxels_masks/subj%01d/roi_1d_mask_subj%02d_floc-bodies.npy" % (args.output_root, subj, subj))

    # mask = np.ones(mask.shape).astype(bool) # overwrite it
    try:  # take out zero voxels
        non_zero_mask = np.load(
            "%s/output/voxels_masks/subj%d/nonzero_voxels_subj%02d.npy"
            % (args.output_root, subj, subj)
        )
        print("Masking zero voxels...")
        mask = mask[non_zero_mask]
    except FileNotFoundError:
        pass

    text_activations = []
    model, _ = clip.load("ViT-B/32", device=device)
    for caption in all_captions:
        with torch.no_grad():
            text = clip.tokenize(caption).to(device)
            text_activations.append(model.encode_text(text).cpu().data.numpy())

    text_activations = np.array(text_activations)

    # print(text_activations.shape)

    weights = np.load(
        "%s/output/encoding_results/subj%d/weights_clip_whole_brain.npy"
        % (args.output_root, subj)
    )
    bias = np.load(
        "%s/output/encoding_results/subj%d/bias_clip_whole_brain.npy"
        % (args.output_root, subj)
    )

    image_activations = get_preloaded_features(
        subj,
        all_coco_ids,
        "clip",
        layer=None,
        features_dir=args.features_dir,
    ).squeeze()

    img_mean = image_activations[train_idx, :].mean(axis=0, keepdims=True)

    clf = get_t2i_map()
    image_activations_hat = clf.predict(text_activations.squeeze())
    print("image activate hat mean: " + str(np.mean(image_activations_hat)))
    print("image mean: " + str(np.mean(img_mean)))

    pred_ceiling = (image_activations[test_idx] - img_mean) @ weights + bias
    pred_t2i = (image_activations_hat - img_mean) @ weights + bias

    print("prediction shape: ")
    print(pred_t2i.shape)
    pred_t2i = pred_t2i.squeeze()[:, mask]
    pred_ceiling = pred_ceiling.squeeze()[:, mask]

    image_gt = np.load(
        "%s/output/encoding_results/subj%s/pred_clip_whole_brain.p"
        % (args.output_root, subj),
        allow_pickle=True,
    )[1][:gt_subset_n, mask]

    from sklearn.metrics import mean_squared_error

    score_func = mean_squared_error
    score_func = r2_score

    scores_t2i = np.zeros((len(image_gt), len(pred_t2i)))
    scores_ceiling = np.zeros((len(image_gt), len(pred_t2i)))

    from numpy import matlib

    for i in tqdm(range(len(image_gt))):
        scores_t2i[i, :] = score_func(
            matlib.repmat(image_gt[i], len(pred_t2i), 1).T, pred_t2i.T
        )
        scores_ceiling[i, :] = score_func(
            matlib.repmat(image_gt[i], len(pred_ceiling), 1).T, pred_ceiling.T
        )

    accs_t2i, prediction_idx_t2i = compute_acc(scores_t2i)
    accs_ceiling, _ = compute_acc(scores_ceiling)

    plt.figure()
    plt.imshow(scores_t2i, aspect="auto")
    plt.colorbar()
    try:
        plt.savefig("figures/captions_decoding/scores_t2i_%s.png" % subj)
    except ValueError:
        pass

    prediction_output = [
        [all_captions[prediction_idx_t2i[i]], all_captions[i]]
        for i in range(gt_subset_n)
    ]

    with open(
        "%s/output/caption_decoding/pred_subj%s.pkl" % (args.output_root, subj), "wb"
    ) as f:
        pickle.dump(prediction_output, f)

    return accs_t2i, accs_ceiling, prediction_output


def decode_captions_for_voxels_i2i(subj):
    all_captions = []
    all_coco_ids = np.load(
        "%s/output/coco_ID_of_repeats_subj%02d.npy" % (args.output_root, subj)
    )

    _, test_idx = train_test_split(
        np.arange(len(all_coco_ids)), test_size=0.15, random_state=42
    )

    test_coco_ids = all_coco_ids[test_idx]
    for cid in tqdm(test_coco_ids):
        captions = load_captions(cid)
        all_captions.append(captions[0])

    test_n = 20
    test_sample_idx = test_idx[:test_n]
    candidate_idx = test_idx[test_n:]
    # print(candidate_idx)

    candidate_caption = all_captions[test_n:]
    test_sample_caption = all_captions[:test_n]

    text_activations = []
    model, _ = clip.load("ViT-B/32", device=device)
    for caption in all_captions:
        with torch.no_grad():
            text = clip.tokenize(caption).to(device)
            text_activations.append(model.encode_text(text).cpu().data.numpy())

    text_activations = np.array(text_activations)
    # print(text_activations.shape)

    image_activations = get_preloaded_features(
        subj,
        all_coco_ids,
        "clip",
        layer=None,
        features_dir=args.features_dir,
    ).squeeze()

    image_test_samples = image_activations[test_sample_idx]
    image_candidates = image_activations[candidate_idx]

    scores = np.zeros((len(image_test_samples), len(image_candidates)))
    from scipy.spatial.distance import cosine

    for t, sample in enumerate(image_test_samples):
        scores[t, :] = [cosine(sample, c) for c in image_candidates]

    plt.figure()
    plt.imshow(scores, aspect="auto")
    plt.colorbar()
    plt.savefig("figures/captions_decoding/scores_i2i_%s.png" % subj)

    prediction_output = [
        [candidate_caption[idx], test_sample_caption[i]]
        for i, idx in enumerate(prediction_idx)
    ]

    with open(
        "%s/output/caption_decoding/pred_subj_out_of_set_%s.pkl"
        % (args.output_root, subj),
        "wb",
    ) as f:
        pickle.dump(prediction_output, f)

    return prediction_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subj", type=int, default=1, help="Specify which subject to build model on."
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/user_data/yuanw3/project_outputs/NSD",
        help="Specify the path to the output directory",
    )
    parser.add_argument(
        "--features_dir",
        type=str,
        default="/user_data/yuanw3/project_outputs/NSD/features",
        help="Specify the path to the features directory",
    )

    args = parser.parse_args()

    accs = []
    subjs = np.arange(1, 9)
    # for subj in np.arange(1, 9):
    for subj in subjs:
        acc_t2i, acc_ceiling, prediction_output = decode_captions_for_voxels_t2i(subj)
        print(prediction_output[20:30])
        accs.append(np.vstack((acc_t2i, acc_ceiling)))

    accs = np.array(accs)
    print(accs.shape)

    plt.figure()
    plt.plot(subjs, accs[0, 0, :], label="top 1")
    plt.plot(subjs, accs[0, 1, :], label="top 5")
    plt.plot(subjs, accs[0, 2, :], label="acc rank")
    plt.legend()
    plt.savefig("figures/captions_decoding/acc_i2t_across_subj.png")

    plt.figure()
    plt.plot(subjs, accs[1, 0, :], label="top 1")
    plt.plot(subjs, accs[1, 1, :], label="top 5")
    plt.plot(subjs, accs[1, 2, :], label="acc rank")
    plt.legend()
    plt.savefig("figures/captions_decoding/acc_i2t_across_subj_testing.png")

    normalized_accs = accs[0, ::] / accs[1, :, :]
    plt.figure()
    plt.plot(subjs, normalized_accs[0, :], label="top 1")
    plt.plot(subjs, normalized_accs[1, :], label="top 5")
    plt.plot(subjs, normalized_accs[2, :], label="acc rank")
    plt.legend()
    plt.savefig("figures/captions_decoding/acc_i2t_across_subj_normalized.png")

    # prediction_output = decode_captions_for_voxels_i2i(args.subj)
    # print(prediction_output)
