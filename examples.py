from statistics import mean

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

    graph = Graph(setting)

    for bs in [ 8, 16, 32, 64]:
        print("\nBS ", bs)

        for strat in [ "random", "v1"]:
            print("Evaluate strat: ", strat)
            acc = []
            for i in range(100):
                setting.configure(batch_size=bs, prediction=strat)
                setting.predict(False)
                acc.append(setting.predictor.acc)

            graph.add_datapoint(str(bs), mean(acc), strat)
    print("done")
    graph.show()
    return setting, graph
