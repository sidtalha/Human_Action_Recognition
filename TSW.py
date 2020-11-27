import yaml
from source import load_data, feature_extraction, protocol, classification, load_feature
import importlib as imp
import numpy as np
import sys

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import pickle
import ruamel.yaml

import torch

from torch.utils.data import DataLoader


# %%

def yaml_loader(file_path):
    with open(file_path, "r") as file_descriptor:
        info_data = yaml.safe_load(file_descriptor)
    return info_data


# %% load yaml

filepath = 'Config1.yaml'
info_data = yaml_loader(filepath)


# %% write yaml

# with open(filepath, 'w') as file:
#     documents = yaml.dump(layer4, file)
#

# %% parameters

par = info_data.get("parameters")
item_load = par['load_data']
item_feature = par['feature_extraction']

# %% load data

exec(open("reload.py").read())


if item_load:

    loading = info_data.get("load_data")
    items = loading['data_inertial']
    Fc = items.get("src")

    path = items.get("data_path")
    dim = items.get("dim")
    nature = items.get("nature")

    df = eval(Fc)(path, dim, nature)

    pickle_out = open("save\data_init","wb")
    pickle.dump(df, pickle_out)
    pickle_out.close()

else:
    pickle_in = open("save\data_init","rb")
    df = pickle.load(pickle_in)


# %% feature extraction


exec(open("reload.py").read())

if item_feature:

    Feat = info_data.get("feature_extraction")

    it = Feat['data_inertial']

    par = np.array([], dtype=object)
    # par = {}
    k = -1
    for j, m in it.items():
        #    k += 1
        par = np.append(par, m)
        # par[k] = m

    src_f = it.get("src")

    # keys={1, 2}
    # par1 = {k: par[k] for k in keys}

    len(par[1:3])

    data, z = eval(src_f)(df.data, par[1], par[2], par[3])


    pickle_out = open("save\\feature", "wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()

else:
    pickle_in = open("save\\feature", "rb")
    data = pickle.load(pickle_in)

# %%

x1 = data['feature'].shape[0]
x2 = data['feature'].shape[1]

nlab = len(np.unique(data['labels'])) + 1


#%%

architecture = info_data.get("Architecture")
x1f = x1
x2f = x2
for i, k in enumerate(architecture.keys()):

    if k[0:4] == 'conv':
        r1 = k[-1]
    elif k[0:6] =='linear':
        r2 = k[-1]


    if k[0:4] == 'conv' or k[0:4] == 'pool':
        layer = architecture[k]
        padding = layer['padding']
        dilation = layer['dilation']
        kernel_size = layer['kernel_size']
        stride = layer['stride']
        x1f = (x1f+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0]+1
        x2f = (x2f+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1]+1




x1f = int(x1f)
x2f = int(x2f)


filepath = 'Config1.yaml'
yaml = ruamel.yaml.YAML()

with open(filepath) as file_descriptor:
    info_data = yaml.load(file_descriptor)

info_data['data_info']['h'] = x1
info_data['data_info']['w'] = x2


info_data['Architecture']['linear'+r2]['out_features'] = nlab


info_data['Architecture']['linear1']['in_features'] = x1f*x2f\
                                        * info_data['Architecture']['conv'+r1]['out_channels']


with open(filepath, 'w') as fp:
    yaml.dump(info_data, fp)


# %% protocol

exec(open("reload.py").read())


X = protocol.proto(data)



#%%


# #
# # dat = data_iter.next()
# #
# # feature, labs = dat
#
# # %% train model
#
# exec(open("reload.py").read())
# filepath = 'Config1.yaml'
# info_data = yaml_loader(filepath)
# pr = info_data.get("training")
# par = np.array([], dtype=object)
# k = -1
# for j, m in pr.items():
#     par = np.append(par, m)
#
# src_c = pr.get("src")
#
# results = eval(src_c)(X["train_data"], X["train_label"], X["test_data"], X["test_label"],
#                       X["actions"], par[1], par[2], par[3], par[4])
#
#
# # %%
# # %pylab
# conf = results['conf']
#
# sm = conf.sum(axis = 1)
#
# cp = sm.reshape((len(sm), 1))*np.ones((1, len(sm)))
#
# df_cm = pd.DataFrame(conf/cp, range(len(conf)), range(len(conf)))
# plt.figure(figsize=(10,7))
# sn.set(font_scale=1.4) # for label size
# sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}) # font size
# plt.show()
#
# print(results['acc'])
#
#


