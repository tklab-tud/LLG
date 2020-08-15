from collections import OrderedDict

from Setting import *
from Graph import *
from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfile


# Experiment 1.1
def visualize_class_prediction_accuracy_vs_batchsize(run=None):
    if run is None:
        run = load_json()
    else:
        run = run.copy()
    graph = Graph("Batch Size", "Prediction Accuracy")
    meta = run["meta"].copy()
    run.__delitem__("meta")

    # Prediction Acc
    for id, run_name in enumerate(run):
        i = id % meta["n"]
        step = id // meta["n"]
        graph.add_datapoint("v2", run[run_name]["prediction_results"]["accuracy"], str(meta["bsrange"][step]))

    graph.plot_line(style="solid", color="Blue")

    graph.show()
    f = asksaveasfile(mode='w',
                      defaultextension=".png",
                      initialfile="prediction_accuracy_vs_batchsize.png",
                      initialdir=run.popitem()[1]["parameter"]["result_path"])
    graph.save_f(f)

# Experiment 1.2
def visualize_flawles_class_prediction_accuracy_vs_batchsize(run=None):
    if run is None:
        run = load_json()
    else:
        run = run.copy()
    graph = Graph("Batch Size", "Perfect predictions")
    meta = run["meta"].copy()
    run.__delitem__("meta")

    # Pefect Prediction
    cnt = {}
    for bs in meta["bsrange"]:
        cnt.update({bs: 0})

    for id, run_name in enumerate(run):
        i = id % meta["n"]
        trainstep = id // meta["n"]
        if run[run_name]["prediction_results"]["accuracy"] == 1.0:
            cnt[run[run_name]["parameter"]["batch_size"]] += 1

    for c in cnt.items():
        graph.add_datapoint(c[0], c[1] / meta["n"])

    graph.plot_bar(color="Blue", alt_ax=False)

    graph.show()
    graph.save_f(asksaveasfile(mode='w',
                               defaultextension=".png",
                               initialfile="prediction_accuracy_vs_training.png",
                               initialdir=run.popitem()[1]["parameter"]["result_path"]))



# Experiment 2
def visualize_class_prediction_accuracy_vs_training(run=None):
    if run is None:
        run = load_json()
    else:
        run = run.copy()
    graph = Graph("Train Samples", "Prediction Accuracy", "Test Acc")
    meta = run["meta"].copy()
    run.__delitem__("meta")

    # Prediction Acc
    for id, run_name in enumerate(run):
        i = id % meta["n"]
        trainstep = id // meta["n"]
        graph.add_datapoint("v2", run[run_name]["prediction_results"]["accuracy"], trainstep)

    graph.plot_line(style="solid", color="Blue", alt_ax=False)
    graph.data = []

    # Test Acc
    for id, run_name in enumerate(run):
        i = id % meta["n"]
        trainstep = id // meta["n"]
        graph.add_datapoint("test_acc", run[run_name]["parameter"]["test_acc"], trainstep)

    graph.plot_line(style="solid", color="Red", alt_ax=True)

    graph.show()
    graph.save_f(asksaveasfile(mode='w',
                               defaultextension=".png",
                               initialfile="prediction_accuracy_vs_training.png",
                               initialdir=run.popitem()[1]["parameter"]["result_path"]))

# Experiment 3: Good Fidelity
# Instead of evaluating class prediction accuracy this experiment performs full recreation attacks.
# It will evaluate the image similarity by plotting the percentage of samples that reach a mse below a threshold.
# The threshold is plotted to the x axis.

def visualize_good_fidelity(run=None):
    if run is None:
        run = load_json()
    else:
        run = run.copy()
    graph = Graph("Fidelity Score", "Percentage of Samples")
    meta = run["meta"].copy()
    run.__delitem__("meta")

    # prepare fidelity
    fidelity = {}
    for strat in meta["strats"]:
        fidelity.update({strat: {}})
        for step in meta["steps"]:
            fidelity[strat].update({step: 0})

    # go through run
    for id, run_name in enumerate(run):

        for step in meta["steps"]:
            for mse in run[run_name]["attack_results"]["mses"][-1]:
                if mse < step:
                    fidelity[run[run_name]["parameter"]["prediction"]][step] += 1

    length = meta["bs"] * meta["n"]

    for i, strat in enumerate(fidelity):
        for step in fidelity[strat]:
            graph.add_datapoint(strat, fidelity[strat][step] / length, str(step))
        style = "solid" if i == 1 else "--"
        graph.plot_line(style=style)
        graph.data = []

    graph.show()
    graph.save_f(asksaveasfile(mode='w',
                               defaultextension=".png",
                               initialfile="good_fidelity.png",
                               initialdir=run.popitem()[1]["parameter"]["result_path"]))




def load_json():
    Tk().withdraw()
    filename = askopenfilename(initialdir="./results", defaultextension='.json',
                               filetypes=[('Json', '*.json')])

    with open(filename) as f:
        dump = OrderedDict(json.load(f))

    return dump
