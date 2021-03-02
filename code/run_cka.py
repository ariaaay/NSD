import argparse
import numpy as np
import matplotlib.pyplot as plt

from cka import cca_core
from CKA.cka import linear_CKA, kernel_CKA

def make_sym_matrix(X):
    X = X + X.T - np.diag(np.diag(X))
    return X

def pairwise_cka(reps):
    lcka = np.zeros((len(reps), len(reps)))
    kcka = lcka.copy()
    for i, rep1 in enumerate(reps[:-1]):
        for k, rep2 in enumerate(reps[i:]):
            lcka[i, k+i] = linear_CKA(rep1, rep2)
            kcka[i, k+i] =  kernel_CKA(rep1, rep2)

    # to make it symmetric
    lcka = make_sym_matrix(lcka)
    kcka = make_sym_matrix(kcka)
    return lcka, kcka


def imshow_cka_results(out, labels, figname):
    n = out.shape[0]

    plt.figure()
    plt.imshow(out)
    plt.yticks(range(0,n), labels=labels)
    plt.colorbar()

    for i in range(n):
        for j in range(n):
            text = plt.text(j, i, round(out[i,j],2), ha="center", va="center", color="w")
    plt.savefig(figname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fearture_dir", type=str, default="/user_data/yuanw3/project_outputs/NSD/features/subj1")
    parser.add_argument("--output_dir", type=str, default="/user_data/yuanw3/project_outputs/NSD/output/cka/subj1")
    parser.add_argument("--figure_dir", type=str, default="/home/yuanw3/NSD/figures/CKA/subj1")
    args = parser.parse_args()

    # experiment 1 - across all layers for a network
    tasks = ["class_1000", "class_places"]
    layers = ["_input_layer1", "_input_layer2", "_input_layer3", "_input_layer5", "", "_output_layer1"]
    layer_labels = layers.copy()
    layer_labels[4] = "_bottle_neck"
    
    for task in tasks:
        reps = list()
        for layer in layers:
            reps.append(np.load("%s/taskrepr_%s%s.npy" % (args.task, layer)))
            lcka, kcka = pairwise_cka(reps)
            np.save("%s/%s_all_layers_linear_cka.npy" % (args.output_dir, task), lcka)
            np.save("%s/%s_all_layers_kernel_cka.npy" % (args.output_dir, task), kcka)

        figname = "%s/%s_linear_cka.png" % (args.figure_dir, task)
        imshow_cka_results(lcka, layer_labels, figname=figname)

        figname = "%s/%s_kernal_cka.png" % (args.figure_dir, task)
        imshow_cka_results(kcka, layer_labels, figname=figname)

    