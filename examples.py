import torch
import time
from Graph import *
from Setting import Setting


def simple_attack():
    setting = Setting(dlg_iterations=30,
                      log_interval=3,
                      batch_size=4,
                      use_seed=False,
                      dlg_lr=0.5,
                      )

    graph = Graph(setting, "Iterations", "MSE")

    setting.attack()
    setting.show_composed_image()
    graph.add_mses()
    graph.show()
    return setting, graph


def prediction_accuracy_vs_batchsize_line(biased=False):
    setting = Setting(log_interval=1,
                      use_seed=False,
                      )

    graph = Prediction_accuracy_graph(setting, "Batch-Size", "Accuracy")

    prediction_string = ""
    maxbs = 33
    reinit = False
    global_id = 0
    setting = Setting(log_interval=1, use_seed=False)


    for bs in range(1, maxbs):
        print("\nBS ", bs)
        if biased:
            target = (bs // 2) * [0] + (bs // 4) * [1]
        else:
            target = []

        for strat in ["v1", "v2"]:

            for i in range(1):

                run_name = "{}{:2.0f}{:2.0f}".format(strat, bs, i)
                if reinit:
                    setting = Setting(log_interval=1, use_seed=False, batch_size=bs, prediction=strat,
                                      target=list(target), run_name=run_name)
                else:
                    setting.configure(batch_size=bs, prediction=strat, target=list(target), run_name=run_name)

                graph.setting = setting
                setting.reset_seeds()
                setting.predict()
                graph.add_prediction_acc(strat, bs)
                grads = setting.predictor.gradients_for_prediction

                prediction_string += strat + "; " + str(i) + "; " + str(global_id) + "; "
                prediction_string += "; ".join(["{0:,.2f}".format(x) for x in grads]) + "; "
                prediction_string += "; " + "{0:,.2f}".format(setting.predictor.acc) + "; "
                prediction_string += "; ".join([str(x) for x in list(setting.predictor.prediction)]) + "; " * (
                        maxbs - setting.parameter["batch_size"])
                origlabels = list(setting.dlg.orig_label)
                origlabels.sort()
                prediction_string += "; ".join([str(x.item()) for x in origlabels])
                prediction_string += "\n"

                global_id += 1

    prediction_string = prediction_string.replace(".", ",")

    if not os.path.exists(setting.parameter["result_path"]):
        os.makedirs(setting.parameter["result_path"])

    with open(setting.parameter["result_path"] + "prediction.csv", "w") as file:
        file.write(prediction_string)

    graph.plot_line()
    graph.save("Accuracy_vs_Batchsize_Biased")
    return setting, graph


def prediction_accuracy_vs_strategie_bar(biased=False):
    setting = Setting(log_interval=5,
                      use_seed=False,
                      )

    graph = Prediction_accuracy_graph(setting, "Batch-Size", "Accuracy")

    bs = 8

    if biased:
        target = (bs // 2) * [0] + (bs // 4) * [1]
    else:
        target = []

    for strat in ["v1", "random", "simplified"]:
        for i in range(2):
            run_name = "{}{:2.0f}".format(strat, i)
            if strat == "dlg":
                setting.configure(improved=False, target=target, batch_size=bs, dlg_iterations=30, run_name=run_name)
            else:
                setting.configure(improved=True, prediction=strat, target=target, batch_size=bs, run_name=run_name)

            setting.predict(True)
            graph.add_prediction_acc(strat, bs)
            setting.store_everything()

    graph.plot_bar()
    graph.save("Accuracy_vs_Batchsize_Biased")
    return setting, graph


def mse_vs_iteration_line(biased=False, bs=1):
    setting = Setting(dlg_iterations=60,
                      log_interval=1,
                      batch_size=bs,
                      use_seed=True,
                      seed=1340,
                      dlg_lr=1,
                      )

    graph = Mses_vs_Iterations_graph(setting, "Iterations", "MSE")

    if biased:
        target = (bs // 2) * [0] + (bs // 4) * [1]
    else:
        target = []

    # idlg strats
    for strat in ["v1", "simplified", "dlg"]:
        setting.reset_seeds()
        for n in range(1):
            run_name = "{}{:2.0f}".format(strat, n)

            if strat == "dlg":
                setting.configure(improved=False, target=target, run_name=run_name)
            else:
                setting.configure(improved=True, prediction=strat, target=target, run_name=run_name)

            print(run_name, setting.ids)
            setting.attack()
            graph.add_all_mses(strat)
            setting.store_everything()
            setting.delete()

    graph.plot_line()
    graph.save("Mses_vs_Iterations")

    return setting, graph


def store_and_load():
    setting, graph = prediction_accuracy_vs_batchsize_line()
    graph.show()
    print("load")
    setting = Setting.load_json(None)
    graph = Prediction_accuracy_graph(setting[0], "Batch-Size", "Accuracy")
    for s in setting:
        graph.setting = s
        graph.add_prediction_acc(s.parameter["prediction"], s.parameter["batch_size"])

    graph.plot_line()
    graph.show()
    print("done")
    return setting, graph


def perfect_prediction_line(biased=False):
    setting = Setting(improved=True, prediction="v1", use_seed=False)
    graph = Prediction_accuracy_graph(setting, "Batch-Size", "Perfect Predictions")
    for bs in range(1, 32):

        if biased:
            target = (bs // 2) * [0] + (bs // 4) * [1]
        else:
            target = []

        n = 100
        cnt = 0
        for i in range(n):
            setting.configure(target=target, batch_size=bs)
            setting.predict()
            if setting.predictor.acc == 1.0:
                cnt += 1

        graph.add_datapoint("v1", cnt / n, bs)

    graph.plot_line()
    graph.show()

    return setting, graph
