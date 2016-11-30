import csv
import numpy as np
from myCNN import *

# Notice: You should change the path to where train.csv is!
with open('Your Path/train.csv', newline='') as f:
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

# Preprocessing the dataset
X = (X-np.mean(X)) / np.std(X)
# We should reshape the dataset to fit the size of the network we build
X = np.reshape(X, [len(X), 1, 28, 28])
Xtrain = X[:4000]
ytrain = y[:4000]
Xtest = X[40000:]
ytest = y[40000:]

# We will build a CNN with structure below:
# Input(28*28*1)
# Centered: Input - mean(Input)
# Conv1: input=28*28*1, 5 filters(3*3*1), filt_size=3, padding=1, stride=3, output=10*10*5
# ReLU1: input=10*10*5, output=10*10*5
# Pool1: input=10*10*5, filt_size=2, stride=2, output=5*5*5
# Conv2(FC): input=5*5*5, 10 filters(5*5*5), filt_size=5, padding=0, stride=1, output=1*1*10
# Softmax: input=1*1*10, output=1*1*10

structure = [('conv',[1,28,28]), ('relu', [5,10,10]), ('pool',[5,10,10]),
             ('conv',[5,5,5]),('softmax', [10,1,1])]
cnn = Network(structure)
filts = [(0,3), (3,5)]
Initialize(cnn, filts)
# set the stride and padding of the 0th and 3rd layer
cnn.take_layer(0).stride = 3
cnn.take_layer(3).padding = 0

# If you have already trained the cnn, and have some trained parameters saved,
# you can write the command below to load the parameters saved:
# Load(cnn, '***.npz')

accuracy = Train(cnn, Xtrain, ytrain, step=0.1, epochs=1)
e = 0
for i in range(5):
    e += 1
    print('epoch:', e)
    accuracy = Train(cnn, Xtrain, ytrain, step=0.1, epochs=1)
Save(cnn, 'CNNparameter.npz')

Test(cnn, Xtest, ytest)
