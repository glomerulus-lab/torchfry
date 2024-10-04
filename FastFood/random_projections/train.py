import numpy as np
import sklearn as sc
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import projections
import argparse
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn_extra.kernel_approximation import Fastfood
from sklearn.svm import LinearSVC



#inputs
parser = argparse.ArgumentParser()
parser.add_argument('-dataset', required=True)
args = parser.parse_args()

#load iris dataset
if args.dataset == "iris":
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

    scaler = StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)

#load mnist data set
if args.dataset == "mnist":
    mnist = fetch_openml('mnist_784')
    x = mnist.data
    y = mnist.target 

    scaler = StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2017)


#Test Params
scale = 20
dimensions = [512, 1024, 2048, 4096]


#RBF kernel
gamma = 1/(2*scale**2)
svc = SVC(kernel='rbf', gamma=gamma, C=10)
svc.fit(x_train, y_train)
test = svc.predict(x_test)
train = svc.predict(x_train)
rbf_acc_test = accuracy_score(test, y_test)
rbf_acc_train = accuracy_score(train,y_train)
print("Kernel: RBF", "Train_Accuracy:", rbf_acc_train, "Test_Accuracy:", rbf_acc_test)
print("--------------------------------------------------------------------------------")


#RKS implementation
rks_acc_test = []
rks_acc_train = []
rks_time = []

for dimension in dimensions:
    #project train
    rks = projections.rks(dimension, scale)
    rks.fit(x_train)
    #transform
    x_train_rks = rks.transform(x_train)
    x_test_rks = rks.transform(x_test)

    #train
    lr = LinearSVC(C=10, dual=False)
    lr.fit(x_train_rks, y_train)
    test = lr.predict(x_test_rks)
    train = lr.predict(x_train_rks)
    test_acc = accuracy_score(test, y_test)
    train_acc = accuracy_score(train,y_train)


    rks_acc_test.append(test_acc)
    rks_acc_train.append(train_acc)
    print("Kernel: RKS", "Train_Accuracy:", train_acc, "Test_Accuracy:", test_acc)
    print("--------------------------------------------------------------------------------")

#fastfood
ff_acc_test = []
ff_acc_train = []
ff_time = []

for dimension in dimensions:
    #project train
    ff = Fastfood(sigma=scale, n_components=dimension)
    ff.fit(x_train)
    #project transform
    x_train_ff = ff.transform(x_train)
    x_test_ff = ff.transform(x_test)

    #train
    lr = LinearSVC(C=10, dual=False)
    lr.fit(x_train_ff, y_train)
    test = lr.predict(x_test_ff)
    train = lr.predict(x_train_ff)
    test_acc = accuracy_score(test, y_test)
    train_acc = accuracy_score(train,y_train)

    ff_acc_test.append(test_acc)
    ff_acc_train.append(train_acc)
    print("Kernel: FF", "Train_Accuracy:", train_acc, "Test_Accuracy:", test_acc)
    print("--------------------------------------------------------------------------------")


np.save("results/dimensions",dimensions)
np.save("results/rbf_acc_train",rbf_acc_train)
np.save("results/rbf_acc_test",rbf_acc_test)
np.save("results/rks_acc_train",rks_acc_train)
np.save("results/rks_acc_test",rks_acc_test)
np.save("results/ff_acc_train",ff_acc_train)
np.save("results/ff_acc_test",ff_acc_test)



                   