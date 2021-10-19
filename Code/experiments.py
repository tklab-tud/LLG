from Graph import *
from Setting import Setting
from train import test
import datetime
import os
import numpy as np
import json
import time
import torch
import sys

result_path = "results/{}/".format(str(datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S"))),


################## Completely Configurable ###################
def experiment(dataloader, list_datasets, list_bs, list_balanced, list_versions, extent, n, trainsize=100, trainsteps=0,
               federated=False, num_users=1, path=None, model="LeNet", store_individual_gradients= False,
               differential_privacy: bool=False, alphas: list=[], noise_multiplier: float=1.0, max_norm: float=1.0, noise_type: str="gauss",
               defenses=[], dropout: float=0.0, compression: bool=False, threshold: float=0.1,
               store_composed_image=False, store_separate_images=False, cuda_id=0, **more_args):
    run = {"meta": {
        "list_datasets": list_datasets,
        "trainsize": trainsize,
        "trainsteps": trainsteps,
        "list_bs": list_bs,
        "list_balanced": list_balanced,
        "list_versions": list_versions,
        "extent": extent,
        "n": n,
        "model": model
    }}

    pre_run_test = False
    if not trainsteps == 0:
        pre_run_test = True

    if len(defenses) == 0:
        defenses.append("none")


    if "dropout" in defenses:
        dropout_save = dropout
        dropout = 0.0

    setting = Setting(dataloader, result_path=path, model=model, cuda_id=cuda_id, differential_privacy=differential_privacy, alphas=alphas,
                      noise_multiplier=noise_multiplier, max_norm=max_norm, noise_type=noise_type, train_size=trainsize,
                      dropout=dropout, compression=compression, threshold=threshold, federated=federated, num_users=num_users)

    progress = 0
    todo = len(list_datasets)* len(list_bs)* len(list_balanced)*len(list_versions)*len(defenses)*n*(trainsteps+1)
    for dataset in list_datasets:
        if dataloader.currently_loaded != dataset:
            dataloader.load_dataset(dataset)

        for trainstep in range(trainsteps+1):

            for bs in list_bs:
                for balanced in list_balanced:
                    for version in list_versions:
                        for defense in defenses:
                            # defense == "none"
                            compression = False
                            differential_privacy = False
                            dropout = 0.0
                            if defense == "compression":
                                compression = True
                            elif defense == "dp":
                                differential_privacy = True
                            elif defense == "dropout":
                                dropout = dropout_save

                            for i in range(n):
                                print("\ti: {:07.0f} / {:07.0f} ".format(progress, todo))
                                progress += 1

                                # The name of the run for later identification and file naming
                                run_name = "{}_{:03.0f}_{}_{}_{}_{}_{}_{:07.0f}".format(
                                    dataset, bs, balanced, version, extent, trainstep, defense, i)

                                # defining attacked batch. Later it will be filled with random samples if len(target) < bs
                                if balanced:
                                    target = []
                                else:
                                    target = dataloader.get_random_targets(bs)

                                # configure the setting
                                setting.configure(dataset=dataset, batch_size=bs, version=version,
                                                  run_name=run_name, targets=target, result_path=path,
                                                  compression=compression,
                                                  differential_privacy=differential_privacy,
                                                  dropout=dropout,
                                                  federated=federated,
                                                  num_users=num_users,
                                                  **more_args)

                                if pre_run_test:
                                    test(setting)
                                    pre_run_test = False

                                keep = trainsteps != 0 and i == n-1 and setting.parameter["local_training"]
                                # run the attack
                                setting.attack(extent, keep)

                                if store_composed_image:
                                    setting.result.store_composed_image() #saves the (i)dlg reconstructed images composed
                                if store_separate_images:
                                    setting.result.store_separate_images() #saves the (i)dlg reconstructed images seperatly
                                if store_composed_image or store_separate_images:
                                    setting.result.delete() #deletes the images in the memory, to safe resources.

                                # dump the current state of the attack
                                run.update({run_name: setting.get_backup(store_individual_gradients)})

            # train the model for trainsize batches (last time needs no training afterwards)
            if trainstep < trainsteps:
                print("\nTrainstep ", trainstep)
                setting.train(trainsize)

    # build file name using parameters
    param_str = ""
    beautify_keys = {
        "train_steps": "global",
        "train_size": "semi",           # unmonitored global
        "local_iterations": "local",
        "v1": "LLG",
        "v2": "LLG+",
        "v3": "LLG*",
        "random": "RND",
        "dlg": "DLG",
    }

    iterations = {
        "train_steps": trainsteps,
        "train_size": trainsize,
        "local_iterations": setting.parameter["local_iterations"],
    }

    key_params = ["dataset", "model", "balanced", "version", "train_steps", "local_iterations", "num_users", "defenses"]
    for key in key_params:
        if key == "balanced":
            for balanced in list_balanced:
                param_str += "_IID" if balanced else "_nonIID"
        elif key == "version":
            if extent != "victim_side":
                for version in list_versions:
                    param_str += "_" + beautify_keys[version]
        elif key == "defenses":
            for defense in defenses:
                if defense == "dp":
                    noise_mult = setting.parameter["noise_multiplier"]
                    max_norm = setting.parameter["max_norm"]
                    if max_norm == None:
                        param_str += "_noise[" + str(noise_mult) + "]"
                    else:
                        param_str += "_dp[" + str(noise_mult) + ", " + str(max_norm) + "]"
                elif defense == "compression":
                    param_str += "_comp[" + str(setting.parameter["threshold"]) + "]"
        elif key in ["train_steps", "train_size", "local_iterations"]:
            if key == "local_iterations" and not setting.parameter["local_training"]:
                continue
            if iterations[key] >= 1:
                param_str += "_" + beautify_keys[key] + "[" + str(iterations[key]*iterations["train_size"]) + "]"
        elif key == "num_users":
            if setting.parameter["federated"]:
                param_str += "_users[" + str(setting.parameter[key]) + "]"
        else:
            param_str += "_" + setting.parameter[key]

    file_name = "dump" + param_str + ".json"

    # write the stored results
    print("dumping results to: " + setting.parameter["result_path"] + file_name)
    dump_to_json(run, setting.parameter["result_path"], file_name)
    return run



################### Help functions ######################

# dump a dictionary of multiple setting-backups into a json file
def dump_to_json(run, path, name):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + name, "w") as file:
        json.dump(run, file)


# set seeds or unset seeds. -1 for random
def set_seeds(seed):
    if seed == -1:
        torch.manual_seed(int(1000 * time.time() % 2 ** 32))
        np.random.seed(int(1000 * time.time() % 2 ** 32))
    else:
        torch.manual_seed(seed)
        np.random.seed(seed)
