from Graph import *
from Setting import Setting
import datetime

result_path = "results/{}/".format(str(datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S"))),


#################### Experiment 1: Prediction Accuracy ####################

def prediction_accuracy_vs_batchsize(n, bsrange, dataset, balanced):
    setting = Setting(log_interval=1,
                      use_seed=False,
                      seed=1337,
                      dataset=dataset,)

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
                target = (bs // 2) * [4] + (bs // 4) * [2]
                target = target[:bs]

            setting.configure(batch_size=bs, prediction="v2", run_name=run_name, targets=target)
            setting.reinit_weights()
            setting.predict()
            graph.add_datapoint("v2", setting.predictor.acc, bs)
            setting.store_json()

            prediction_string += "v2;" + str(i) + ";" + str(global_id)
            prediction_string += "; " + "{0:,.2f}".format(setting.predictor.acc) + "; "
            prediction_string += "; ".join([str(x) for x in list(setting.predictor.prediction)]) + "; " * (
                    max(bsrange) - setting.parameter["batch_size"])
            origlabels = list(setting.parameter["orig_label"])
            origlabels.sort()
            prediction_string +=";" + "; ".join([str(x.item()) for x in origlabels])
            prediction_string += "\n"

            global_id += 1

    prediction_string = prediction_string.replace(".", ",")

    if not os.path.exists(setting.parameter["result_path"]):
        os.makedirs(setting.parameter["result_path"])

    with open(setting.parameter["result_path"] + "prediction.csv", "w") as file:
        file.write(prediction_string)

    graph.plot_line()
    graph.save(setting.parameter["result_path"], "Accuracy_vs_Batchsize.png")
    return setting, graph




######################## old code ###################################
"""
def simple_attack():
    setting = Setting(dlg_iterations=30,
                      log_interval=3,
                      batch_size=4,
                      use_seed=False,
                      dlg_lr=0.5,
                      improved=False

                      )

    setting.attack()
    setting.show_composed_image()

    return setting


def prediction_accuracy_vs_batchsize_line(biased=False):
    setting = Setting(log_interval=1,
                      use_seed=False,
                      )
    graph = Prediction_accuracy_graph(setting, "Batch-Size", "Accuracy")
    maxbs = 129
    reinit = False
    global_id = 0
    prediction_string = "Strat; #try; #glo;"
    for i in range(100):
        prediction_string += "grad_" + str(i) + ";"
    prediction_string += "Acc;Prediction"
    for i in range(1, maxbs):
        prediction_string += ";"
    prediction_string += "Original\n"

    for dataset in ["MNIST", "CIFAR"]:
        for bs in range(100, 102):
            print("\nBS ", bs)
            if biased:
                target = (bs // 2) * [4] + (bs // 4) * [2]
            else:
                target = []


            for strat in ["v2"]:

                for i in range(1):

                    run_name = "{}{:2.0f}{:2.0f}".format(strat, bs, i)
                    if reinit:
                        setting = Setting(log_interval=1, use_seed=False, batch_size=bs, prediction=strat,
                                          targets=list(target), run_name=run_name, dataset=dataset,
                                          dataloader=setting.dataloader, result_path=result_path)
                        setting.dataloader.setting = setting
                    else:
                        setting.configure(batch_size=bs, prediction=strat, targets=list(target), run_name=run_name,
                                          dataset=dataset)

                    graph.setting = setting
                    setting.reset_seeds()
                    setting.predict()
                    graph.add_prediction_acc(strat+"_"+dataset, bs)
                    setting.store_json()


                    grads = setting.predictor.gradients_for_prediction
                    prediction_string += strat + "; " + str(i) + "; " + str(global_id) + "; "
                    prediction_string += "; ".join(["{0:,.4f}".format(x) for x in grads])
                    if strat == "random": prediction_string += "; " * (100 - 1)
                    prediction_string += "; " + "{0:,.2f}".format(setting.predictor.acc) + "; "
                    prediction_string += "; ".join([str(x) for x in list(setting.predictor.prediction)]) + "; " * (
                            maxbs - setting.parameter["batch_size"])
                    origlabels = list(setting.parameter["orig_label"])
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
    graph.save("Accuracy_vs_Batchsize")
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

    for strat in ["v1", "random", "idlg"]:
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
    setting = Setting(dlg_iterations=20,
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
    n = 0

    ids = np.random.randint(0, len(setting.train_dataset), setting.parameter["batch_size"])
    ids = [x.item() for x in ids]
    for strat in [ "v2", "idlg", "dlg"]:
        run_name = "_{}_{:3.0f}".format(strat, n)
        n += 1
        print(run_name)

        if strat == "dlg":
            setting.configure(improved=False, target=target, run_name=run_name, ids=ids)
        else:
            setting.configure(improved=True, prediction=strat, target=target, run_name=run_name, ids=ids)

        setting.attack()
        graph.add_all_mses(strat)
        setting.store_data()
        setting.delete()

        graph.plot_line()
        graph.show()

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
    setting = Setting(use_seed=False)
    graph = Prediction_accuracy_graph(setting, "Batch-Size", "Perfect Predictions")
    for bs in range(1, 32):

        if biased:
            target = (bs // 2) * [0] + (bs // 4) * [1]
        else:
            target = []

        n = 10
        for strat in ["v1", "v2"]:
            print("bs ", bs, "strat ", strat)
            cnt = 0
            for i in range(n):
                setting.configure(target=target, batch_size=bs, prediction=strat)
                setting.predict()
                if setting.predictor.acc == 1.0:
                    cnt += 1

            graph.add_datapoint(strat, cnt / n, bs)

    graph.plot_line()
    graph.show()

    return setting, graph


def mse_vs_batchsize_line(biased=False, iterations=30):
    setting = Setting(dlg_iterations=iterations,
                      log_interval=1,
                      use_seed=True,
                      seed=1340,
                      dlg_lr=1,
                      )

    graph = Graph(setting, "Iterations", "MSE")

    for bs in [16, 32, 64]:
        if biased:
            target = (bs // 2) * [0] + (bs // 4) * [1]
        else:
            target = []

        for strat in ["v2", "idlg", "dlg"]:
            setting.reset_seeds()
            run_name = "{} {}".format(strat, bs)
            print(run_name)

            if strat == "dlg":
                setting.configure(improved=False, target=target, run_name=run_name, batch_size=bs)
            else:
                setting.configure(improved=True, prediction=strat, target=target, run_name=run_name, batch_size=bs)

            setting.attack()
            graph.add_datapoint(strat, np.mean(setting.result.mses, 0)[-1], bs)
            setting.store_data()
            setting.delete()

    graph.plot_line()
    graph.show()
    graph.save("Mses_vs_Batchsize")

    return setting, graph



def load_with_styles():
    setting = Setting.load_json(None)
    graph = Mses_vs_Iterations_graph(setting[0], "Batch-Size", "Accuracy")

    i = 0
    for s in setting:
        graph.setting = s
        graph.add_datapoint(s.parameter["prediction"], s.predictor.false, s.parameter["batch_size"])


    style = [(0, (1, 10)), (0, (1, 1)), (0, (1, 1)), (0, (5, 10)), (0, (5, 5)), (0, (5, 1)), (0, (3, 10, 1, 10)),
             (0, (3, 5, 1, 5)), (0, (3, 1, 1, 1)), (0, (3, 5, 1, 5, 1, 5)), (0, (3, 10, 1, 10, 1, 10)),
             (0, (3, 1, 1, 1, 1, 1))]
    for i_l, label in enumerate(dict.fromkeys([label for (label, _, _) in graph.data])):
        style = style[i_l % 12]
        graph.plot_line(style=style, clear=False, color=color_map(label))
        graph.data = []
        i += 1

    #graph.subplot.set_ylim(0, 0.5)
    #graph.subplot.set_xlim(0, 10)

    graph.show()
    print("done")
    return setting, graph
    
"""


def color_map(label):
    return {
        "v1": 'b',
        "v2": 'g',
        "v3": 'c',
        "v4": 'y',
        "v5": 'k',
        "dlg": 'r',
        "idlg": 'm',
    }[label]

