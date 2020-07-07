from Setting import Setting
import time


def main():
    ############## Build your attack here ######################

    setting = Setting(dlg_iterations=1,
                      log_interval=1,
                      use_seed=True,
                      batch_size=4,
                      prediction="v1",
                      dlg_lr=0.5
                      )

    for bs in [4, 8, 16, 32]:
        print("\nBS: ", bs)
        s= int(time.time()*1000)%2**32
        for strat in ["random", "v1", "simplified"]:
            setting.configure(batch_size=bs, prediction=strat, seed=s)
            setting.predict(True)

    ############################################################


if __name__ == '__main__':
    main()
    print("Run finished")
