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
    [x] Make attack configurable
    [x] Make prediction use Settings in order to allow better configuration, e.g. call accuracy

Todo:
    [ ] Visualize mse, loss
    [ ] Poison Model to increase accuracy
    [ ] Machine Learning as prediction strategy
    [ ] Further prediction strategies



# Build your attack based on these elements:

    ## Minimal example
    ```python
    import Setting

    setting = Setting()
    setting.setting.attack()
    setting.store_everything()
    ```



    ## Parameter
    Adjust the settings at init or with configure:
    ```python
    setting = Setting(dlg_iterations=30, log_interval=3 )
    setting.configure(dlg_iterations=30, log_interval=3)
     ```
    ###dataset
    ###batch_size
    ###model
    ###log_interval
    ###use_seed
    ###seed
    ###result_path
    ###run_name,
    ###dlg_lr
    ###dlg_iterations
    ###prediction
    ###improved
    ###lr
    ###epochs
    ###max_epoch_size
    ###test_size


    ## Functions
    Settings provides a whole lot of functions to experiment with

    ###configure
    ###restore_default_parameter
    ###reset_seeds
    ###load_dataset
    ###print_parameter
    ###pretrain
    ###predict
    ###attack
    ###store_everything
    ###store_composed_image
    ###store_separate_images
    ###store_data
    ###show_composed_image
    ###delete



    ## Iteration
    Attacks can be performed iterative. You might change settings between the runs. If you are trying to create files be sure to adjust run_name or result_path in every iteration or files will be overwritten.
    ```python
    setting = Setting(dlg_iterations=30,
                      log_interval=3,
                      use_seed=True,
                      target=[1, 2, 3, 4, 5, 6, 7, 8, 9]
                      )

    for i in [1, 2, 4, 8]:
        setting.configure(run_name=str(i), batch_size=i)
        setting.print_parameter()
        setting.attack()
        setting.show_composed_image()
        setting.store_composed_image()
        setting.store_data()
        setting.store_separate_images()
        setting.delete()
    ```