import pandas as pd
import numpy as np
import math
import random


def sigmoid(y):
    return 1 / (1 + math.exp(-y))


def delta_base(t, y):
    try:
        return -(t - sigmoid(y)) * sigmoid(y) * (1 - sigmoid(y))
    except:
        return 0


MAX_EPOCH = 10000
LEARNING_RATE = 0.01
CHECK_POINT_EPOCH = 1

data = pd.read_csv("data.csv")

print("\n\n===========data===========")
print(data)
print("===========data===========\n\n")

p = 0
w = np.array([[random.normalvariate(0, 1) for i in range(3)]])
error_sum = 0
correct_answers_num_sum = 0
accuracy_list = []
error_list = []
for i in range(MAX_EPOCH):
    epoch = i + 1
    correct_answers_num = 0
    for index, row in data.iterrows():
        x = np.array([[1, row["w_num"], row["b_num"]]])
        y = np.dot(w, x.reshape(3, 1))
        t = 0 if row["target"] == "B" else 1

        delta = np.array(
            [[delta_base(t, y), x[0][1]*delta_base(t, y), x[0][2]*delta_base(t, y)]])
        w = w - LEARNING_RATE*delta
        error_sum += (t - sigmoid(y)) ** 2 / 2

        if sigmoid(y) <= 0.5 and row["target"] == "B":
            correct_answers_num += 1
            continue

        if sigmoid(y) > 0.5 and row["target"] == "W":
            correct_answers_num += 1
            continue

    if epoch % CHECK_POINT_EPOCH == 0:
        error_avarage = error_sum/len(data)*100/CHECK_POINT_EPOCH
        error_list.append([epoch, error_avarage])
        error_sum = 0

    correct_answers_num_sum += correct_answers_num
    if epoch % CHECK_POINT_EPOCH == 0:
        accuracy_avarage = correct_answers_num_sum / \
            len(data)*100/CHECK_POINT_EPOCH
        accuracy_list.append([epoch, accuracy_avarage])
        correct_answers_num_sum = 0

accuracy_list = np.array(accuracy_list)
error_list = np.array(error_list)
np.savetxt("delta_accuracy.csv", accuracy_list, delimiter=",")
np.savetxt("delta_error.csv", error_list, delimiter=",")
