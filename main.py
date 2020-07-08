import examples


def main():
    ############## Build your attack here ######################

    setting, graph = examples.accuracy_test()
    graph.save(setting.parameter["result_path"]+"acc_test.jpg")

    ############################################################


if __name__ == '__main__':
    main()
    print("Run finished")
