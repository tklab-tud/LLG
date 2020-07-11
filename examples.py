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
    setting = Setting(log_interval=1,
                      use_seed=False,
                      )

    graph = Prediction_accuracy_graph(setting, "Batch-Size", "Accuracy")

    for bs in range(1, 33):
        print("\nBS ", bs)
        if biased:
            target = (bs // 2) * [0] + (bs // 4) * [1]
        else:
            target = []

        for strat in ["v1", "random"]:
            for i in range(100):
                setting.configure(batch_size=bs, prediction=strat, target=list(target))
                setting.predict()
                graph.add_prediction_acc(strat, bs)

    graph.plot_line()
    graph.save("Accuracy_vs_Batchsize_Biased")
    return setting, graph


def prediction_accuracy_vs_strategie_bar(biased=False):
    setting = Setting(log_interval=1,
                      use_seed=False,
                      )

    graph = Prediction_accuracy_graph(setting, "Batch-Size", "Accuracy")

    bs = 8

    if biased:
        target = (bs // 2) * [0] + (bs // 4) * [1]
    else:
        target = []

    for strat in ["v1", "random"]:
        for i in range(100):
            setting.configure(batch_size=bs, prediction=strat, target=list(target))
            setting.predict()
            graph.add_prediction_acc(strat, bs)

    graph.plot_bar()
    graph.save("Accuracy_vs_Batchsize_Biased")
    return setting, graph


def mse_vs_iteration_line(biased=False, bs=1):
    setting = Setting(dlg_iterations=5,
                      log_interval=1,
                      batch_size=bs,
                      use_seed=False,
                      dlg_lr=0.3,
                      )

    graph = Mses_vs_Iterations_graph(setting, "Iterations", "MSE")

    if biased:
        target = (bs // 2) * [0] + (bs // 4) * [1]
    else:
        target = []

    # idlg strats
    for strat in ["v1" ]:
        for n in range(10):
            print(strat, n)
            if strat == "dlg":
                setting.configure(improved=False, target=target)
            else:
                setting.configure(prediction=strat, target=target)

            setting.attack()
            graph.add_all_mses(strat)

            if setting.result.mses.mean()> 3:
                pdb.set_trace()

            setting.delete()

    graph.plot_line()
    graph.save("Mses_vs_Iterations")

    return setting, graph
