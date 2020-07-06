from Setting import Setting


def main():
    ############## Build your attack here ######################

    setting = Setting(dlg_iterations=50,
                      log_interval=5,
                      use_seed=True,
                      batch_size=8,
                      prediction="random"
                      )

    for i in [4,8,16,32,64,128,256]:
        half = i // 2
        for pred in ["random", "v1", "simplified"]:
            setting.configure(run_name=str("_1"),
                              target=list([1]*half+[2,3,4,5,6,7,8,9]*half),
                              batch_size=i,
                              prediction=pred
                              )
            print("{}-Prediction:".format(pred))
            setting.predict()



    ############################################################


if __name__ == '__main__':
    main()
    print("Run finished")
