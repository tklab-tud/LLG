from Graph import *
from Setting import Setting
import pdb


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
    setting = Setting(log_interval=5,
                      use_seed=False,
                      )

    graph = Prediction_accuracy_graph(setting, "Batch-Size", "Accuracy")

    for bs in range(1, 8):
        print("\nBS ", bs)
        if biased:
            target = (bs // 2) * [0] + (bs // 4) * [1]
        else:
            target = []

        for strat in ["v1", "random"]:
            for i in range(10):
                run_name = "{}{:2.0f}{:2.0f}".format(strat,bs, i )
                setting.configure(batch_size=bs, prediction=strat, target=list(target), run_name=run_name)
                setting.predict()
                graph.add_prediction_acc(strat, bs)
                setting.store_everything()

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
    setting = Setting(dlg_iterations=20,
                      log_interval=1,
                      batch_size=bs,
                      use_seed=True,
                      seed=1337,
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
        for n in range(3):
            run_name = "{}{:2.0f}".format(strat,n)

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