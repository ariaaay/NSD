import pickle
import numpy as np

def load_data(model, task, output_root, subj=1, measure="corr"):
    if task is None:
        output = pickle.load(
            open(
                "%s/output/encoding_results/subj%d/%s_%s_whole_brain.p"
                % (output_root, subj, measure, model),
                "rb",
            )
        )
    else:
        output = pickle.load(
            open(
                "%s/output/encoding_results/subj%d/%s_%s_%s_whole_brain.p"
                % (output_root, subj, measure, model, task),
                "rb",
            )
        )
    if measure == "corr":
        out = np.array(output)[:, 0]
    else:
        out = np.array(output)
    return out