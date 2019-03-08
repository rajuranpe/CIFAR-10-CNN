import keras
import utils
import createNetwork

model = utils.loadModel()
createNetwork.runNetwork(model, 64, 10)
