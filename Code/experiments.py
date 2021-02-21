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
def experiment(dataloader, list_datasets, list_bs, list_balanced, list_versions, extent, n, trainsize=100, trainsteps=0, path=None, model="LeNet", store_individual_gradients= False,
               differential_privacy: bool=False, alphas: list=[], noise_multiplier: float=1.0, max_norm: float=1.0, noise_type: str="gauss",
               defenses=[], dropout: float=0.0, compression: bool=False, threshold: float=0.1,
               store_composed_image=False, store_separate_images=False, **more_args):
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

    if len(defenses) == 0:
        defenses.append("none")

    if "dropout" in defenses:
        dropout_save = dropout
        dropout = 0.0

    setting = Setting(dataloader, result_path=path, model=model, differential_privacy=differential_privacy, alphas=alphas,
                      noise_multiplier=noise_multiplier, max_norm=max_norm, noise_type=noise_type,
                      dropout=dropout, compression=compression, threshold=threshold)

    progress = 0
    todo = len(list_datasets)* len(list_bs)* len(list_balanced)*len(list_versions)*len(defenses)*n*(trainsteps+1)
    for dataset in list_datasets:
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
                                    # we define unbalanced as 50% class a, 25% class b, 25% random
                                    choice1 = np.random.choice(range(setting.parameter["num_classes"])).item()
                                    choice2 = np.random.choice(
                                        np.setdiff1d(range(setting.parameter["num_classes"]), choice1)).item()
                                    target = (bs // 2) * [choice1] + (bs // 4) * [choice2]
                                    target = target[:bs]

                                # configure the setting
                                setting.configure(dataset=dataset, batch_size=bs, version=version,
                                                  run_name=run_name, targets=target, result_path=path,
                                                  compression=compression,
                                                  differential_privacy=differential_privacy,
                                                  dropout=dropout,
                                                  **more_args)

                                if not trainsteps == 0:
                                    test(setting)

                                # run the attack
                                setting.attack(extent)

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

    # write the stored results
    print("dumping results to: " + setting.parameter["result_path"] + "dump.json")
    dump_to_json(run, setting.parameter["result_path"], "dump.json")
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
