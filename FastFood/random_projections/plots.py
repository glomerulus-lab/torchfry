import matplotlib.pyplot as plt
import csv
import numpy as np

dimensions = np.load("results/dimensions.npy")
rbf_acc_train = np.load("results/rbf_acc_train.npy")
rbf_acc_test = np.load("results/rbf_acc_test.npy")
rks_acc_train = np.load("results/rks_acc_train.npy")
rks_acc_test = np.load("results/rks_acc_test.npy")
ff_acc_train = np.load("results/ff_acc_train.npy")
ff_acc_test = np.load("results/ff_acc_test.npy")

#plot for test set accuracy
plt.figure()
x=[dimensions[0], dimensions[-1]]
y=[rbf_acc_test,rbf_acc_test]
plt.plot(x,y, label='RBF_Test')
plt.plot(dimensions, rks_acc_test, label='RKS_Test', marker='o')
plt.plot(dimensions, ff_acc_test, label='FF_Train', marker='o')
plt.xlabel('Dimension (n)')
plt.ylabel('Accuracy')
plt.title('Projection Accuracy (Test)')
plt.legend()
plt.savefig('graphs/test_acc.png', bbox_inches='tight')

#plot for train set accuracy
plt.figure()
x=[dimensions[0], dimensions[-1]]
y=[rbf_acc_train,rbf_acc_train]
plt.plot(x,y, label='RBF_Train')
plt.plot(dimensions, rks_acc_train, label='RKS_Train', marker='o')
plt.plot(dimensions, ff_acc_train, label='FF_Train', marker='o')
plt.xlabel('Dimension (n)')
plt.ylabel('Accuracy')
plt.title('Projection Accuracy (Train)')
plt.legend()
plt.savefig('graphs/train_acc.png', bbox_inches='tight')
