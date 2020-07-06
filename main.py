from Setting import Setting


def main():
    ############## Build your attack here ######################

    setting = Setting()

    for i in [1, 2, 4, 8]:
        setting.configure(batch_size=i)
        setting.target([3, 5, 2, 6])
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
