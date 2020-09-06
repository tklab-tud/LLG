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
def experiment(dataloader, list_datasets, list_bs, list_balanced, list_versions, extent, n, trainsize=100, trainsteps=0, path=None, reconstruction_steps=0):
    run = {"meta": {
        "list_datasets": list_datasets,
        "trainsize": trainsize,
        "trainsteps": trainsteps,
        "list_bs": list_bs,
        "list_balanced": list_balanced,
        "list_versions": list_versions,
        "extent": extent,
        "n": n,
        "reconstruction_steps": reconstruction_steps
    }}

    setting = Setting(dataloader, result_path=path)

    progress = 0
    todo = len(list_datasets)* len(list_bs)* len(list_balanced)*len(list_versions)*n*trainsteps
    for dataset in list_datasets:
        for trainstep in range(trainsteps+1):
            if not trainsteps == 0:
                test(setting)

            for bs in list_bs:
                for balanced in list_balanced:
                    for version in list_versions:
                        for i in range(n):
                            print("\ti: {:07.0f} / {:07.0f} ".format(progress, todo))
                            progress += 1

                            # The name of the run for later identification and file naming
                            run_name = "{}_{:03.0f}_{}_{}_{}_{}_{:07.0f}".format(
                                dataset, bs, balanced, version, extent, trainstep, i)

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
                                              dlg_iterations=reconstruction_steps)

                            # run the attack
                            setting.attack(extent)

                            # dump the current state of the attack
                            run.update({run_name: setting.get_backup()})

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
