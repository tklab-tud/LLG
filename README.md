# Further improved deep leakage from gradients

This framework extends the iDLG Attack from
[iDLG: Improved Deep Leakage from Gradients](https://arxiv.org/pdf/2001.02610.pdf) by Zhao et Al.

It is build upon the authors [implementation](https://github.com/PatrickZH/Improved-Deep-Leakage-from-Gradients)

As the iDLG attack states it "works with a simplified scenario of sharing gradients of every datum. In other words, iDLG can identify the ground-truth labels only if gradients w.r.t. every sample in a training batch are provided."

In other words this attack does not work with batch sizes other than 1.
In their workaround they split the victim's batch into single samples. Those samples get evaluated by their label prediction method.
In the further steps they recreate images from the predicted labels and the gradients of the pooled batch.

This is an unrealistic scenario. The concept of having batch sizes other than 1 inherits to pool the gradients (e.g. by calculating the mean value or the sum) during the learning process.
The result is used to update the model or in case of federated learning might be shared to the server. Gradients of single samples are not shared.

This framework tries to fill this gap with the exploration of different label prediction strategies for batch sizes larger than 1.


It adds additional features:
    [x] Batch_Size configurable
    [x] Different Prediction strategies
    [x] Interchangable Models
    [x] Data dump as json
    [x] Image visualisation and storage
    [x] Pretrain model
    [x] Specific targets, balanced/unbalanced Data
    [x] Loop your attack and create a run
    [x] Configure details of your attack
    [x] Visualization
    [x] Dump loader
    [x] Image creaetion from dump

Todo:
    [ ] Clean up code & doc
	[ ] Finish report
	[ ] Clearify full run / fast run selection


Optional:
    [ ] Poison Model to increase accuracy
    [ ] Further prediction strategie ML
	[ ] Analyse gradients of second last layer before they are added together as identifying values



Attacks can be build in main.py by creating a Setting, configuring it, starting the attack and storing it as json.

You can do a fast run, which will only do the prediction or a full run which does the prediction and recreates images using idlg/dlg afterwards.

Or you can just load up the experiments from experiments.py and visualise their result. 