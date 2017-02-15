#!/usr/bin/python

import json
import random
import pickle
import time
import math
import copy
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

def tree_mae_calc(node, df):
    acc = 0.0
    total_len = len(df.index)

    for _, row in df.iterrows():
        acc += abs(tree_eval(node, row) - row[node.data.class_var])

    return acc / total_len

def tree_rmse_calc(node, df):
    acc = 0.0
    total_len = len(df.index)

    for _, row in df.iterrows():
        acc += math.pow((tree_eval(node, row) - row[node.data.class_var]), 2)

    return math.sqrt(acc / total_len)


def cxval_k_folds_split(df, k_folds, seed):

    random.seed(seed)

    dataframes = []
    group_size = int(round(df.shape[0]*(1.0/k_folds)))

    for i in range(k_folds-1):
        rows = random.sample(list(df.index), group_size)
        dataframes.append(df.ix[rows])
        df = df.drop(rows)

    dataframes.append(df)

    return dataframes


def cxval_select_fold(i_fold, df_folds):
    df_folds_copy = copy.deepcopy(df_folds)

    if 0 <= i_fold < len(df_folds):
        test_df = df_folds_copy[i_fold]
        del df_folds_copy[i_fold]
        train_df = pd.concat(df_folds_copy)
        return train_df, test_df

    else:
        raise Exception('Group not in range!')

def cxval_test(df, class_var, var_desc, leaf_size, k_folds=5, seed=1):
    df_folds = cxval_k_folds_split(df, k_folds, seed)
    rmse_results = []
    mae_results = []

    for i in range(k_folds):
        train_df, test_df = cxval_select_fold(i, df_folds)
        tree = tree_trainer(train_df, class_var, var_desc, leaf_size)

        mae_results.append(tree_mae_calc(tree, test_df))
        rmse_results.append(tree_rmse_calc(tree, test_df))

    return sum(mae_results)/k_folds, sum(rmse_results)/k_folds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AeroMetTree model training command line utility.')
    parser.add_argument('--data', dest='data_path', help='Path to the CSV file containing the data to train the tree', required=True)
    parser.add_argument('--config', dest='config_path', help='Path to the JSON file containing the parameters to set model', required=True)

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

    data_files = ["datasets/eddt_clean.csv", "datasets/egll_clean.csv", "datasets/lebl_clean.csv", "datasets/lfpg_clean.csv", "datasets/limc_clean.csv", "datasets/yssy_clean.csv", "datasets/zbaa_clean.csv"]
    for data_file in data_files:
        print(data_file)
        df = pd.read_csv(data_file)
        with open(args.config_path) as conf_file:    
            tree_params = json.load(conf_file)

            class_var = tree_params['output']
            # 1 Classic
            tree_desc = {}

            """
            for var in tree_params['input']:
                if not tree_params['contiguous_splits'] and var['type'] == "cir":
                    tree_desc[var['name']] = {"type": var['type'], "method": "subset", "bounds": [[-np.inf, np.inf]]}
                else:
                    tree_desc[var['name']] = {"type": var['type'], "method": "classic", "bounds": [[-np.inf, np.inf]]}
            """

            for size in [1000, 500, 250, 100, 50]:
                print("MaxLeafSize: ", size)
                
                #Linear
                for var in tree_params['input']:
                    tree_desc[var['name']] = {"type": "lin", "method": "classic", "bounds": [[-np.inf, np.inf]]}

                _, a = cxval_test(df, class_var, tree_desc, size)
                print("  - Linear: ", a, end='')
                
		#Lund
                for var in tree_params['input']:
                    if var['type'] == "cir":
                        tree_desc[var['name']] = {"type": var['type'], "method": "subset", "bounds": [[-np.inf, np.inf]]}

                _, a = cxval_test(df, class_var, tree_desc, size)
                print("    Lund: ", a, end='')
                
		#Lund
                for var in tree_params['input']:
                    if var['type'] == "cir":
                        tree_desc[var['name']] = {"type": var['type'], "method": "classic", "bounds": [[-np.inf, np.inf]]}

                _, a = cxval_test(df, class_var, tree_desc, size)
                print("    Our: ", a)

                """
                tree = tree_trainer(df, class_var, tree_desc, size)
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
                """
