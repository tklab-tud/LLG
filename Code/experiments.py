from Graph import *
from Setting import Setting
import datetime
import os
import numpy as np
import json
import time
import torch

result_path = "results/{}/".format(str(datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S"))),


#################### Experiment 1: Class Prediction Accuracy ####################
# This will run the class prediction for a range of batch sizes each n times.
# The result can be visualized by:
# visualise_flawless_class_prediction
# visualize_class_prediction_accuracy_vs_batchsize(

def class_prediction_accuracy_vs_batchsize(n, bsrange, dataset, balanced, version):
    run = {"meta": {
        "n": n,
        "bsrange": bsrange,
        "dataset": dataset,
        "balanced": balanced,
        "version": version
    }}
    setting = Setting(log_interval=1,
                      dataset=dataset, )


    for bs in bsrange:
        print("\nBS ", bs)
        for i in range(n):
            run_name = "{}_{:3.0f}_{:3.0f}".format(version, bs, i)

            if balanced:
                target = []
            else:
                choice1 = np.random.choice(range(setting.parameter["num_classes"])).item()
                choice2 = np.random.choice(np.setdiff1d(range(setting.parameter["num_classes"]), choice1)).item()
                target = (bs // 2) * [choice1] + (bs // 4) * [choice2]
                target = target[:bs]

            setting.configure(batch_size=bs, prediction=version, run_name=run_name, targets=target)
            setting.reinit_weights()
            setting.predict()
            run.update({run_name: setting.get_backup()})


    dump_to_json(run, setting.parameter["result_path"], "pred_acc_vs_bs")

    return setting



#################### Experiment 2: Training ####################

def class_prediction_accuracy_vs_training(n, bs, dataset, balanced, trainsize, trainsteps, version):
    run = {"meta": {
        "n": n,
        "bs": bs,
        "dataset": dataset,
        "balanced": balanced,
        "trainsize": trainsize,
        "trainsteps": trainsteps,
        "version": version
    }}

    setting = Setting(log_interval=1,
                      dataset=dataset, )

    for trainstep in range(trainsteps):
        # 1: Evaluate
        for i in range(n):
            run_name = "{}_{:3.0f}_{:3.0f}".format(version, trainstep, i)

            if balanced:
                target = []
            else:
                choice1 = np.random.choice(range(setting.parameter["num_classes"])).item()
                choice2 = np.random.choice(np.setdiff1d(range(setting.parameter["num_classes"]), choice1)).item()
                target = (bs // 2) * [choice1] + (bs // 4) * [choice2]
                target = target[:bs]

            setting.configure(batch_size=bs, prediction=version, run_name=run_name, targets=target)
            setting.predict()
            run.update({run_name: setting.get_backup()})

        # 2: Train
        print("\nTrainstep ", trainstep)
        setting.train(trainsize)

    dump_to_json(run, setting.parameter["result_path"], "pred_acc_vs_training")
    return setting

#################### Experiment 3: good fidelity ####################

def good_fidelity(n, bs, iterations, dataset, balanced):
    steps = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    strats = ["dlg", "v1", "v2", "idlg"]

    run = {"meta": {
        "n": n,
        "bs": bs,
        "iterations": iterations,
        "dataset": dataset,
        "balanced": balanced,
        "strats": strats,
        "steps": steps
    }}

    # set_seeds(0)
    setting = Setting(log_interval=5,
                      dataset=dataset,
                      dlg_iterations=iterations,
                      batch_size=bs, )

    # Prepare empty fidelity dictionary

    fidelity = {}
    for strat in strats:
        fidelity.update({strat: {}})
        for step in steps:
            fidelity[strat].update({step: 0})

    for i in range(n):
        # Choosing target should use random seed.
        set_seeds(-1)
        if balanced:
            target = np.random.randint(0, setting.parameter["num_classes"], bs).tolist()
        else:
            choice1 = np.random.choice(range(setting.parameter["num_classes"])).item()
            choice2 = np.random.choice(np.setdiff1d(range(setting.parameter["num_classes"]), choice1)).item()
            target = (bs // 2) * [choice1] + (bs // 4) * [choice2]
            target.extend(np.random.randint(0, setting.parameter["num_classes"], bs - len(target)).tolist())

            target = target[:bs]

        # To make the runs comparable we use the same seed for each run.
        seed_for_runs = np.random.randint(2 ^ 32)

        for strat in strats:
            set_seeds(seed_for_runs)
            run_name = "{:3.0f}_{}".format(i, strat)

            setting.configure(prediction=strat, run_name=run_name, targets=target)
            setting.reinit_weights()
            setting.attack()

            setting.result.store_composed_image()
            run.update({run_name: setting.get_backup()})

        setting.result.delete()


    dump_to_json(run, setting.parameter["result_path"], "fidelity")

    return setting


################### Help functions ######################

# dump a dictionary of multiple setting-backups into a json file
def dump_to_json(run, path, name):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + "data{}.json".format(name), "w") as file:
        json.dump(run, file)


# set seeds or unset seeds. -1 for random
def set_seeds(seed):
    if seed == -1:
        torch.manual_seed(int(1000 * time.time() % 2 ** 32))
        np.random.seed(int(1000 * time.time() % 2 ** 32))
    else:
        torch.manual_seed(seed)
        np.random.seed(seed)
