from train_model import *

freq = 1046.5
model = build_model()
train_model(freq, model)
evaluate(freq, model)

