# i²DLG - Indeed improved deep leakage from gradients

This framework implements the iDLG Attack from
[iDLG: Improved Deep Leakage from Gradients](https://arxiv.org/pdf/2001.02610.pdf) by Zhao et Al.

It is build upon the authors [implementation](https://github.com/PatrickZH/Improved-Deep-Leakage-from-Gradients)

As the iDLG attack states it "works with a simplified scenario of sharing gradients of every datum. In other words, iDLG can identify the ground-truth labels only if gradients w.r.t. every sample in a training batch are provided."

In other words this attack does not work with batch sizes other than 1. It fails to predict the labels from an intercepted gradient.
In their workaround they assume the victim will evaluate its data not as batch but one by one and just give away the resulting gradients.

This framework tries to fill this gap with the exploration of different label prediction strategies.


It adds additional features:
    [x] Batch_Size configurable
    [x] Different Prediction strategies
    [x] Different Models
    [x] Data dump as json
    [x] Image visualisation and storage
    [x] Pretrain model
    [x] Specific targets
    [x] Make attack iterable

Todo:
    [ ] Visualize mse, loss
    [ ] Poison Model to increase accuracy
    [ ] Machine Learning as prediction strategy
    [ ] Further prediction strategies


# Build your attack based on these elements:

    ...



