import numpy as np
import matplotlib.pyplot as plt

from util.data_util import load_data

corr_i = load_data("clip", None, ".", subj=1)
corr_t = load_data("clip_text", None, ".", subj=1)

plt.scatter(corr_i, corr_t, alpha=0.05)
plt.plot([-0.1, 1], [-0.1, 1], "r")
plt.xlabel("image")
plt.ylabel("text")
plt.savefig("figures/CLIP/image_vs_text_acc.png")

