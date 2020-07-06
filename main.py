from Setting import Setting


def main():
    ############## Build your attack here ######################

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

    ############################################################


if __name__ == '__main__':
    main()
    print("Run finished")
