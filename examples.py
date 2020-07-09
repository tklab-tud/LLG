from statistics import mean

import numpy as np

from Graph import Graph
from Setting import Setting


def accuracy_test():
    setting = Setting(dlg_iterations=1,
                      log_interval=1,
                      use_seed=False,
                      batch_size=4,
                      prediction="v1",
                      dlg_lr=0.5,
                      # target=list(range(0,9))+[0, 1]*256
                      )

    graph = Graph(setting, "Batch-Size", "Accuracy")

    for bs in [8, 16, 32, 64, 128]:
        print("\nBS ", bs)

        for strat in ["random", "v1"]:
            print("Evaluate strat: ", strat)
            acc = []
            for i in range(100):
                setting.configure(batch_size=bs, prediction=strat)
                setting.predict(False)
                acc.append(setting.predictor.acc)

            graph.add_datapoint(str(bs), mean(acc), strat)
    print("done")
    return setting, graph


def mse_test():
    setting = Setting(dlg_iterations=10,
                      log_interval=2,
                      batch_size=4,
                      use_seed=False,
                      dlg_lr=0.5,
                      )

    graph = Graph(setting, "Iterations", "MSE")
    n = 5

    # dlg
    setting.configure(improved=False)
    mses_list = []
    for i in range(n):
        setting.attack()
        mses = setting.result.mses
        for i_m, step in enumerate(mses):
            mses[i_m] = np.mean(step)
        mses_list.append(mses)

    data = [0] * len(mses)

    for i_s in range(len(mses)):
        for i_n in range(1):
            data[i_s] += mses_list[i_n][i_s][0] / n

    li = setting.parameter["log_interval"]
    graph.add_datarow(list(range(li, len(data) * li + 1, li)), data, "dlg")

    # idlg random/v1/simplified
    for strat in ["v1", "simplified", "random"]:
        setting.configure(prediction=strat, run_name=strat)
        mses_list = []
        for i in range(n):
            setting.attack()
            mses = setting.result.mses
            for i_m, step in enumerate(mses):
                mses[i_m] = np.mean(step)
            mses_list.append(mses)

        data = [0]*len(mses)

        for i_s in range(len(mses)):
            for i_n in range(n):
                data[i_s] += mses_list[i_n][i_s][0]/n

        li =setting.parameter["log_interval"]
        graph.add_datarow(list(range(li, len(data)*li+1, li)), data, strat)


    print("done")
    return setting, graph
