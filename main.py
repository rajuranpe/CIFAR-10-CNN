import keras
import utils
import createNetwork

choice = input("Type \"n\" to create a new network or \"t\" to continue training an existing one: ")
if choice == "n":
    createNetwork.createNetwork()

elif choice == "t":
    while True:
        try:
            epochs = int(input("How many epochs? Recommended 5-30 "))
            if epochs > 0:
                break
            else:
                print("Please give a positive whole number.")
        except ValueError:
            print("Please give a positive whole number.")

    model = utils.loadModel()
    createNetwork.runNetwork(model, 64, epochs)

else:
    print("Please read the instruction message.")
