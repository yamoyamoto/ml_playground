import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier

data = pd.read_csv("data.csv")
train_X = data.drop("target", axis=1)

target_encoder = LabelEncoder()
train_Y = target_encoder.fit_transform(data["target"])

print(train_X)
print(train_Y)

mlp_clf = MLPClassifier(hidden_layer_sizes=(50, 50), random_state=42)
mlp_clf.fit(train_X, train_Y)

score = mlp_clf.score(train_X, train_Y)
loss_curve = mlp_clf.loss_curve_

plt.plot(loss_curve)
plt.xlabel('Iteration')
plt.ylabel('loss')
plt.grid(True)
plt.show()