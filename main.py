#!/Users/pablo/anaconda/envs/python3/bin/python

import json
import pickle
import time
import os.path
import pandas as pd
import numpy as np
import argparse

from data.data import Data
from node.node import Node

def tree_pprinter(node):
    fmtr = ""

    def pprinter(anode):
        nonlocal fmtr

        if anode.split_var is not None:
            print("({}) {} {}".format(len(anode.data.df.index), anode.split_var,
                                       anode.data.var_desc[anode.split_var]['bounds']))

        else:
            print("({}) Leaf {} {}".format(len(anode.data.df.index),
                                    np.var(anode.data.df[anode.data.class_var].values),
                                           ["{} {}".format(key, anode.data.var_desc[key]['bounds']) for key in anode.data.var_desc.keys() if anode.data.var_desc[key]['bounds'] != [[-np.inf, np.inf]]]))

        if anode.left_child is not None:
            print("{} `--".format(fmtr), end="")
            fmtr += "  | "
            pprinter(anode.left_child)
            fmtr = fmtr[:-4]

            print("{} `--".format(fmtr), end="")
            fmtr += "    "
            pprinter(anode.right_child)
            fmtr = fmtr[:-4]

    return pprinter(node)


def tree_trainer(df, class_var, var_desc, stop=50, variance=.01):
        if class_var not in df.columns:
            raise Exception('Class variable not in DataFrame')

        data = Data(df, class_var, var_desc)

        node = Node(data, stop=stop, variance=variance)
        node.split()

        return node

if __name__ == "__main__":

    df = pd.read_csv("./datasets/egll_clean.csv")
    tree = None

    if os.path.isfile("model.save"):
        tree = pickle.load(open("model.save", "rb"))
        tree_pprinter(tree)
        exit(1)

    with open('input.json') as conf_file:    
        tree_params = json.load(conf_file)
        print(tree_params)

        class_var = tree_params['output']
        # 1 Classic
        tree_desc = {}
        for var in tree_params['input']:
            tree_desc[var['name']] = {"type": var['type'], "method": "classic", "bounds": [[-np.inf, np.inf]]}
    print(tree_desc)
    tree = tree_trainer(df, class_var, tree_desc)
    tree_pprinter(tree)
    pickle.dump(tree, open("model.save", "wb"))
