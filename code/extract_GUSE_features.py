from absl import logging

import tensorflow as tf

import tensorflow_hub as hub
import numpy as np
import argparse
from tqdm import tqdm

from util.coco_utils import load_captions

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"  # @param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
model = hub.load(module_url)
print("module %s loaded" % module_url)


def embed(input):
    return model(input)


# Reduce logging output.
logging.set_verbosity(logging.ERROR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subj", default=1, type=int)
    parser.add_argument(
        "--feature_dir",
        type=str,
        default="/user_data/yuanw3/project_outputs/NSD/features",
    )
    parser.add_argument(
        "--project_output_dir",
        type=str,
        default="/user_data/yuanw3/project_outputs/NSD/output",
    )

    args = parser.parse_args()

    stimuli_dir = "/lab_data/tarrlab/common/datasets/NSD_images/images"
    all_coco_ids = np.load(
        "%s/coco_ID_of_repeats_subj%02d.npy" % (args.project_output_dir, args.subj)
    )
    feature_output_dir = "%s/subj%01d" % (args.feature_dir, args.subj)
    all_images_paths = ["%s/%s.jpg" % (stimuli_dir, id) for id in all_coco_ids]
    all_text_features = []
    for cid in tqdm(all_coco_ids):
        captions = load_captions(cid)
        embeddings = embed(captions).numpy()
        embedding_avg = np.mean(embeddings, axis=0)
        all_text_features.append(embedding_avg)

    np.save("%s/GUSE.npy" % feature_output_dir, np.array(all_text_features))
