#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 17:51:59 2018

@author: shubham
"""
import numpy as np
import matplotlib.pyplot as plt
from svm_smo import SVM
import my_data_utils as ut

print("Training SVM Using SMO on synthetic data.")

# Seed random generator functions
np.random.seed(20)

#create synthetic samples
X, Y = ut.syn_samples(ut.line,[0,8],[-8,8],20,20)

np.random.seed(21)
X_test, Y_test = ut.syn_samples(ut.line,[0,8],[-8,8],20,20)

#Initialize SVM
C = 1.0
kernel = "linear"

svm = SVM(C=C,kernel=kernel)

#pass dataset to the svm
svm.feed_data(X,Y)

#plot training dataset
print("Training Data Plot: Red = + 1 / Blue = - 1")
ut.plot_2D_dataset(X,Y)

if svm.kernel_name == "linear":
    print("Initial Wts:",svm.W)
    print("Initial Bias:",svm.b)

dummy = input("\n\nPress Enter to Start training...")
svm.train()
print("===========Training over=======")

if svm.kernel_name == "linear":
    print("Wts after training:", svm.W)
    print("Bias after training:", svm.b)


dummy = input("\n\nPress Enter to Start prediction...")

print ("\nTraining Scores..")
Yp = svm.predict(X)
print(svm.compute_scores(Y,Yp))

print ("\nTest Scores..")
Yp_test = svm.predict(X_test)
print(svm.compute_scores(Y_test,Yp_test))

print ("Training Data Plot")
ut.plot_decision_boundary(svm, X, Y,step=0.01, title="Training Data Plot")

print ("Testing Data Plot")
ut.plot_decision_boundary(svm, X_test, Y_test,step=0.01, title="Test Dataset Plot")

