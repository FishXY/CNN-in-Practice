import csv
import numpy as np
from myCNN import *

# with open('E:/课程/数学课/统计学习/kaggle/train.csv', newline='') as f:
#     train_reader = csv.reader(f)
#     Xset=[]
#     Yset=[]
#     n = 0
#     for line in train_reader:
#         if n == 0:
#             n += 1
#             continue
#         Yset.append(int(line[0]))
#         Xset.append([int(i) for i in line[1:]])

#Datas in test.csv are not labelled, perhaps it is the test datas for the website, not for users
#test=file('/public/home/xieyupku/Downloads/test.csv','rb')
#test_reader = csv.reader(test)

# X = np.array(Xset[:400]) # the origin offset is 40000, and I change it to 4
# y = np.array(Yset[:400])
# Xtest = np.array(Xset[41900:]) # the origin is 40000, total number is 42000
# Ytest = np.array(Yset[41900:])

with open('E:/课程/数学课/统计学习/kaggle/train.csv', newline='') as f:
    train_reader = csv.reader(f)
    X=[]
    y=[]
    n = 0
    for line in train_reader:
        if n == 0:
            n += 1
            continue
        y.append(int(line[0]))
        X.append([int(i) for i in line[1:]])
        n += 1
        if n>500:
            break

# Preprocessing the dataset
X = (X - np.mean(X)) / np.std(X)
# Xtest = (Xtest-np.mean(Xtest)) / np.std(Xtest)

structure = [('conv',[1,28,28]), ('relu', [5,10,10]), ('pool',[5,10,10]),
             ('conv',[5,5,5]),('softmax', [10,1,1])]
cnn = Network(structure)
filts = [(0,3), (3,5)]
Initialize(cnn, filts)
cnn.take_layer(0).stride = 3
cnn.take_layer(3).padding = 0
X = np.reshape(X, [len(X), 1, 28, 28])

accuracy = Train(cnn, X, y, step=1, epochs=5)
e = 5
while accuracy <= 0.99:
    print(accuracy)
    e += 1
    print('epoch:', e)
    Train(cnn, X, y, step=0.1, epochs=1)



# Input(28\*28\*1)
# Centered: Input - mean(Input)
# Conv1: input=28\*28\*1, 5 filters(3\*3\*1), filt_size=3, padding=1, stride=3, output=10\*10\*5
# ReLU1: input=10\*10\*5, output=10\*10\*5
# Pool1: input=10\*10\*5, filt_size=2, stride=2, output=5\*5\*5
# Conv2(FC): input=5\*5\*5, 10 filters(5\*5\*5), filt_size=5, padding=0, stride=1, output=1\*1\*10
# Softmax: input=1\*1\*10, output=1\*1\*10

