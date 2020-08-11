from Graph import *
from Setting import Setting
import datetime
import os
import numpy as np
import json

result_path = "results/{}/".format(str(datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S"))),


#################### Experiment 1: Prediction Accuracy ####################

def prediction_accuracy_vs_batchsize(n, bsrange, dataset, balanced):
    run = {}
    setting = Setting(log_interval=1,
                      use_seed=False,
                      seed=1337,
                      dataset=dataset, )

    graph = Graph("Batchsize", "Prediction Accuracy")

    global_id = 0
    prediction_string = "Strat; #try; #glo; Acc; Prediction"
    for _ in range(max(bsrange)):
        prediction_string += ";"
    prediction_string += "Original\n"

    for bs in bsrange:
        print("\nBS ", bs)
        for i in range(n):
            run_name = "{}_{:3.0f}_{:3.0f}".format("v2", bs, i)

            if balanced:
                target = []
            else:
                choice1 = np.random.choice(range(setting.parameter["num_classes"])).item()
                choice2 = np.random.choice(np.setdiff1d(range(setting.parameter["num_classes"]), choice1)).item()
                target = (bs // 2) * [choice1] + (bs // 4) * [choice2]
                target = target[:bs]

            setting.configure(batch_size=bs, prediction="v2", run_name=run_name, targets=target)
            setting.reinit_weights()
            setting.predict()
            graph.add_datapoint("v2", setting.predictor.acc, bs)
            run.update({run_name: setting.get_backup()})

            prediction_string += "v2;" + str(i) + ";" + str(global_id)
            prediction_string += "; " + "{0:,.2f}".format(setting.predictor.acc) + "; "
            prediction_string += "; ".join([str(x) for x in list(setting.predictor.prediction)]) + "; " * (
                    max(bsrange) - setting.parameter["batch_size"])
            origlabels = list(setting.parameter["orig_label"])
            origlabels.sort()
            prediction_string += ";" + "; ".join([str(x.item()) for x in origlabels])
            prediction_string += "\n"

            global_id += 1

    prediction_string = prediction_string.replace(".", ",")

    if not os.path.exists(setting.parameter["result_path"]):
        os.makedirs(setting.parameter["result_path"])

    with open(setting.parameter["result_path"] + "prediction.csv", "w") as file:
        file.write(prediction_string)

    graph.plot_line()
    graph.save(setting.parameter["result_path"], "Accuracy_vs_Batchsize.png")

    dump_to_json(run, setting.parameter["result_path"], "pred_acc_vs_bs")

    return setting, graph


#################### Experiment 2: good fidelity ####################

def good_fidelity(n, bs, iterations, dataset, balanced):
    run = {}
    setting = Setting(log_interval=5,
                      use_seed=False,
                      seed=1337,
                      dataset=dataset,
                      dlg_iterations=iterations,
                      batch_size=bs, )

    graph = Graph("Fidelity score", "Percentage of samples")

    #Prepare empty fidelity dictionary
    steps = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    strats = ["dlg", "v2", "idlg"]
    fidelity = {}
    for strat in strats:
        fidelity.update({strat : {}})
        for step in steps:
            fidelity[strat].update({step: 0})

    for i in range(n):

        if balanced:
            target = np.random.randint(0, setting.parameter["num_classes"], bs).tolist()
        else:
            choice1 = np.random.choice(range(setting.parameter["num_classes"])).item()
            choice2 = np.random.choice(np.setdiff1d(range(setting.parameter["num_classes"]), choice1)).item()
            target = (bs // 2) * [choice1] + (bs // 4) * [choice2]
            target.extend(np.random.randint(0, setting.parameter["num_classes"], bs - len(target)).tolist())

            target = target[:bs]


        for strat in strats:

            run_name = "{:3.0f}_{}".format(i, strat)

            setting.configure(prediction=strat, run_name=run_name, targets=target)
            setting.reinit_weights()

            setting.attack()

            setting.result.store_composed_image()
            run.update({run_name: setting.get_backup()})

            for step in steps:
                for snap in setting.result.mses:
                    for mse in snap:
                        if mse < step:
                            fidelity[strat][step] += 1

        setting.result.delete()

    length = iterations * bs * n

    for strat in fidelity:
        for step in fidelity[strat]:
            graph.add_datapoint(strat, fidelity[strat][step]/ length, str(step))


    graph.plot_line()
    graph.save(setting.parameter["result_path"], "fidelity.png")

    dump_to_json(run, setting.parameter["result_path"], "fidelity")


    return setting, graph


#################### Bonus: Training ####################

def prediction_accuracy_vs_training(n, bs, dataset, balanced, trainsize, trainsteps):
    run = {}

    setting = Setting(log_interval=1,
                      use_seed=False,
                      seed=1337,
                      dataset=dataset, )

    graph = Graph("Train Samples", "Prediction Accuracy")

    global_id = 0
    prediction_string = "Trainstep; #try; #glo; Acc; Prediction"
    for _ in range(bs):
        prediction_string += ";"
    prediction_string += "Original\n"

    for trainstep in range(trainsteps):
        print("\nTrainstep ", trainstep)

        # training
        setting.train(trainsize)

        for i in range(n):
            run_name = "{}_{:3.0f}_{:3.0f}".format("v2", trainstep, i)

            if balanced:
                target = []
            else:
                choice1 = np.random.choice(range(setting.parameter["num_classes"])).item()
                choice2 = np.random.choice(np.setdiff1d(range(setting.parameter["num_classes"]), choice1)).item()
                target = (bs // 2) * [choice1] + (bs // 4) * [choice2]
                target = target[:bs]

            setting.configure(batch_size=bs, prediction="v2", run_name=run_name, targets=target)
            setting.predict()
            graph.add_datapoint("v2", setting.predictor.acc, trainstep)
            run.update({run_name: setting.get_backup()})

            prediction_string += str(trainstep) + ";" + str(i) + ";" + str(global_id)
            prediction_string += "; " + "{0:,.2f}".format(setting.predictor.acc) + "; "
            prediction_string += "; ".join([str(x) for x in list(setting.predictor.prediction)]) + "; " * bs
            origlabels = list(setting.parameter["orig_label"])
            origlabels.sort()
            prediction_string += ";" + "; ".join([str(x.item()) for x in origlabels])
            prediction_string += "\n"

            global_id += 1

    prediction_string = prediction_string.replace(".", ",")

    if not os.path.exists(setting.parameter["result_path"]):
        os.makedirs(setting.parameter["result_path"])

    with open(setting.parameter["result_path"] + "prediction.csv", "w") as file:
        file.write(prediction_string)

    graph.plot_line()
    graph.save(setting.parameter["result_path"], "Accuracy_vs_Training.png")

    dump_to_json(run, setting.parameter["result_path"], "pred_acc_vs_training")
    return setting, graph


#################### Bonus: Perfect Prediction ####################

def perfect_prediction(n, bsrange, dataset, balanced):
    run = {}
    setting = Setting(log_interval=1,
                      use_seed=False,
                      seed=1337,
                      dataset=dataset, )
    graph = Graph("Batch-Size", "Perfect Predictions")
    for bs in bsrange:
        print("BS: ", bs)

        cnt = 0
        for i in range(n):
            runname = str(bs) + "_" + str(i)
            if balanced:
                target = []
            else:
                choice1 = np.random.choice(range(setting.parameter["num_classes"])).item()
                choice2 = np.random.choice(np.setdiff1d(range(setting.parameter["num_classes"]), choice1)).item()
                target = (bs // 2) * [choice1] + (bs // 4) * [choice2]
                target = target[:bs]

            setting.configure(targets=target, batch_size=bs, prediction="v2", run_name=runname)
            setting.reinit_weights()
            setting.predict()
            run.update({runname: setting.get_backup()})
            if setting.predictor.acc == 1.0:
                cnt += 1

        graph.add_datapoint(bs, cnt / n)

    graph.plot_bar()
    graph.save(setting.parameter["result_path"], "prefect_pred.png")
    graph.show()

    dump_to_json(run, setting.parameter["result_path"], "perf_pred")

    return setting, graph


################### Help functions ######################

# dump a dictionary of multiple setting-backups into a json file
def dump_to_json(run, path, name):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + "data{}.json".format(name), "w") as file:
        json.dump(run, file)
