import pandas as pd
import numpy as np


MAX_EPOCH = 1000
LEARNING_RATE = 0.00001

data = pd.read_csv("data.csv")

print("\n\n===========data===========")
print(data)
print("===========data===========\n\n")

p = 0
w = np.array([[0, 1, -1]])
correct_answers_num_sum = 0
accuracy_list = []
for i in range(MAX_EPOCH):
  epoch = i + 1
  correct_answers_num = 0
  for index, row in data.iterrows():
    x = np.array([[1, row["w_num"], row["b_num"]]])
    output = np.dot(w, x.reshape(3, 1))

    if output <= 0 and row["target"] == "W":
      delta = LEARNING_RATE * x
      w = w + delta
      p = 0
      continue

    if output > 0 and row["target"] == "B":
      delta = LEARNING_RATE * x
      w = w - delta
      p = 0
      continue

    p += 1
    correct_answers_num += 1
    if p == len(data):
      print("連続{0}回正解しました！".format(len(data)))
      break
  
  if epoch % 50 == 0:
    accuracy_avarage = correct_answers_num_sum/len(data)/50*100
    print("epoch: {0} ==> 正解率(平均): {1}%".format(epoch, accuracy_avarage))
    accuracy_list.append([epoch, accuracy_avarage])
    correct_answers_num_sum = 0
  else:
    correct_answers_num_sum += correct_answers_num

accuracy_list = np.array(accuracy_list)
np.savetxt("accuracy.csv", accuracy_list, delimiter=",")
