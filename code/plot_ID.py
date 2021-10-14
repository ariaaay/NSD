import json
import numpy as np
import matplotlib.pyplot as plt
from util.model_config import *

output_dir = "../Cats/figures/IDs"
ID_dict = json.load(open("../Cats/outputs/ID_dict.json", "r"))
# model_type = ["alexnet", "taskrepr"]
task_type = {}
task_type["taskrepr"] = ["class_1000", "class_places", "edge2d", "edge3d"]
task_type["alexnet"] = ["convnet", "place"]
layers_type = {}
layers_type["alexnet"] = conv_layers + fc_layers
layers_type["taskrepr"] = ["input_layer" + str(i) for i in [1, 2, 3, 5]]
layers_type["taskrepr"] += ["output_layer1"]

m = "alexnet"
plt.figure()
for t in task_type[m]:
    ID_means, ID_std = [], []
    for layer in layers_type[m]:
        name = "%s_%s_%s_avgpool.npy" % (t, m, layer)
        ID_means.append(ID_dict[name][0])
        ID_std.append(ID_dict[name][1])
    plt.errorbar(x=np.arange(len(ID_means)), y=ID_means, yerr=ID_std, label=t)
plt.ylabel("Intrinsic Dimension")
plt.xlabel("Layers")
plt.legend()
plt.savefig("%s/%s.png" % (output_dir, m))

m = "taskrepr"
plt.figure()
for t in task_type[m]:
    ID_means, ID_std = [], []
    for layer in layers_type[m]:
        name = "%s_%s_%s.npy" % (m, t, layer)
        try:
            ID_means.append(ID_dict[name][0])
            ID_std.append(ID_dict[name][1])
        except KeyError:
            print(name)
            continue
    plt.errorbar(x=np.arange(len(ID_means)), y=ID_means, yerr=ID_std, label=t)
plt.ylabel("Intrinsic Dimension")
plt.xlabel("Layers")
plt.legend()
plt.savefig("%s/%s.png" % (output_dir, m))
