#!/usr/bin/python

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


def tree_trainer(df, class_var, var_desc, stop=50, variance=.05):
        if class_var not in df.columns:
            raise Exception('Class variable not in DataFrame')

        data = Data(df, class_var, var_desc)

        node = Node(data, stop=stop, variance=variance)
        node.split()

        return node


def is_in_bounds(bounds, value):
    for bound in bounds:
        if bound[0] > bound[1]:
            if bound[0] <= value <= 360.0 or 0.0 <= value < bound[1]:
                return True
        elif bound[0] == 0.0 and value == 360:
            return True
        else:
            if bound[0] <= value < bound[1]:
                return True

    return False


def tree_eval(node, row):
    result = None

    def eval(node, row):
        nonlocal result

        if node.split_var is None:
            result = np.mean(node.data.df[node.data.class_var].values)

        else:
            if is_in_bounds(node.left_child.data.var_desc[node.split_var]["bounds"], row[node.split_var]):
                eval(node.left_child, row)
            elif is_in_bounds(node.right_child.data.var_desc[node.split_var]["bounds"], row[node.split_var]):
                eval(node.right_child, row)
            else:
                print(node.data.var_desc[node.split_var]["bounds"], row[node.split_var])
                print(node.left_child.data.var_desc[node.split_var]["bounds"], row[node.split_var])
                print(node.right_child.data.var_desc[node.split_var]["bounds"], row[node.split_var])
                sys.exit("tree_eval() problem with bounds ????")

    eval(node, row)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AeroMetTree model training command line utility.')
    parser.add_argument('--data', dest='data_path', help='Path to the CSV file containing the data to train the model')
    parser.add_argument('--config', dest='config_path', help='Path to the CSV file containing the data to train the model')

    args = parser.parse_args()
    #print(args.data_path)

    df = pd.read_csv(args.data_path)
    tree = None
    model_name = os.path.splitext(os.path.basename(args.config_path))[0] + ".save"

    """
    if os.path.isfile(model_name):
        tree = pickle.load(open(model_name, "rb"))
        tree_pprinter(tree)
        exit(1)
    """

    with open(args.config_path) as conf_file:    
        tree_params = json.load(conf_file)
        #print(tree_params)

        class_var = tree_params['output']
        # 1 Classic
        tree_desc = {}
        for var in tree_params['input']:
            tree_desc[var['name']] = {"type": var['type'], "method": "classic", "bounds": [[-np.inf, np.inf]]}
   
    for size in [1000, 500, 250, 100, 50]:
        #print(tree_desc)
        #tree = tree_trainer(df, class_var, tree_desc, tree_params['max_leaf_size'])
        tree = tree_trainer(df, class_var, tree_desc, size)
        #tree_pprinter(tree)
        pickle.dump(tree, open(model_name, "wb"))

        acc_tree = 0.0
        acc_gfs = 0.0
        for _, row in df.iterrows():
            acc_tree += (row[tree.data.class_var] - tree_eval(tree, row)) ** 2 
            acc_gfs += (row[tree.data.class_var] - row["gfs_wind_spd"]) ** 2 
       
        print("max_leaf_size", size)
        print("Abs Tree error:", (acc_tree/len(df.index)) ** .5)
        print("Abs GFS error", (acc_gfs/len(df.index)) ** .5)
        print("___________")
