import pandas as pd
import numpy as np
import math

def sigmoid(y):
  return 1 / (1 + math.exp(y))

def g(w0, w1, x1, w2, x2):
  return w0 + w1*x1 + w2*x2

def f(y):
  return 1 / (1+math.exp(y))

def delta_base(t, y):
  return (t - f(y)) * (-math.exp(y)) / ((1 + math.exp(y))**2)

MAX_EPOCH = 10000
LEARNING_RATE = 0.001

data = pd.read_csv("data.csv")

print("\n\n===========data===========")
print(data)
print("===========data===========\n\n")

p = 0
w = np.array([[0, -2, 4]])
error_sum = 0
correct_answers_num_sum = 0
accuracy_list = []
for i in range(MAX_EPOCH):
  epoch = i + 1
  correct_answers_num = 0
  for index, row in data.iterrows():
    x = np.array([[1, row["w_num"], row["b_num"]]])
    y = np.dot(w, x.reshape(3, 1))
    t = 0 if row["target"] == "W" else 1
    delta = np.array([[w[0][0]*delta_base(t, y), w[0][1]*delta_base(t, y), w[0][2]*delta_base(t, y)]])
    w = w - LEARNING_RATE*delta
    # print("delta: ", w)
    # print("Error: ", (t - f(y)) ** 2 / 2)
    error_sum += (t - f(y)) ** 2 / 2

    if y <= 0 and row["target"] == "W":
      # print("{0}番目のデータで正解!".format(index))
      correct_answers_num += 1
      continue

    if y > 0 and row["target"] == "B":
      # print("{0}番目のデータで正解!".format(index))
      correct_answers_num += 1
      continue

  if epoch % 50 == 0:
    error_avarage = error_sum/len(data)/50*100
    print("epoch: {0} ==> 誤差平均: {1},  w:{2}".format(epoch, error_avarage, w))
    accuracy_list.append([epoch, error_avarage])
    error_sum = 0
  else:
    error_sum += correct_answers_num

  if epoch % 50 == 0:
    accuracy_avarage = correct_answers_num_sum/len(data)/50*100
    print("epoch: {0} ==> 正解率(平均): {1}%".format(epoch, accuracy_avarage))
    accuracy_list.append([epoch, accuracy_avarage])
    correct_answers_num_sum = 0
  else:
    correct_answers_num_sum += correct_answers_num

accuracy_list = np.array(accuracy_list)
np.savetxt("accuracy.csv", accuracy_list, delimiter=",")

