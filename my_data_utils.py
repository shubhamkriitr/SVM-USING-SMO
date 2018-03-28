#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 14:55:34 2018

@author: shubham
"""
import numpy as np
import matplotlib.pyplot as plt

def syn_samples (dec_fn,x1_range,x2_range,Np,Nn,dtype=np.float32,sep=5):
    npv=0
    nn=0
    eps = 1e-7
    X = np.zeros(shape=(Np+Nn,2),dtype=dtype)
    Y = np.zeros(shape=(Np+Nn,),dtype=dtype)
    while npv!=Np or nn!=Nn:
        x1 = np.random.uniform(low=x1_range[0], high=x1_range[1])
        x2 = np.random.uniform(low=x2_range[0], high=x2_range[1])
        op = dec_fn(x1,x2)
        #print("for x1={} and X2={}; op={}".format(x1,x2,op))
        if op > eps + sep/2 and npv<Np:
            X[npv,0] = x1
            X[npv,1] = x2
            Y[npv] = 1
            npv+=1
        elif op < eps - sep/2 and nn<Nn:
            X[nn+Np,0] = x1
            X[nn+Np,1] = x2
            Y[nn+Np] = -1
            nn+=1
        #print("npv={} nn={}".format(npv,nn))
    return (X,Y)

def circle (x1,x2,c = [0,0],r = 4):
    return (x1-c[0])*(x1-c[0]) + (x2-c[1])*(x2-c[1]) - r*r

def line (x1,x2,m=2,c=-5):
    return x2-m*x1-c

def x_partition (x1, x2, m =1,cx1=4):
    a = m * (x1 - cx1)
    return (x2-a) * (x2 + a)

def xx_partition (x1, x2, m1=0.1, m2=2, cx1=4, cx2 =4):
    return x2*x_partition(x1,x2,m1,cx1)*x_partition(x1, x2, m2, cx2)



def plot_2D_dataset(X,Y):
    for i in range(X.shape[0]):
        c = "blue"
        if Y[i] == 1:
            c = "red"
        plt.scatter(X[i,0],X[i,1],c=c,alpha=0.5)
    plt.show()

def plot_decision_boundary(h, X, Y,step=0.1,x1_range=None,x2_range=None,title=""):
    """
    Args:
        h(class:'function'): hypothesis (Model)
        X: input dataset (Also Used for determining ranges if xi_range=None)
        Y: output dataset (Shoud have only 1 and -1 as element values)
        step: step size to use for creating mesh-grid
    """
    if x1_range is None and x2_range is None:
        x1r = (X[:,0].min(), X[:,0].max())
        x2r = (X[:,1].min(), X[:,1].max())
    elif (x1_range is not None) and (x2_range is not None):
        x1r = x1_range
        x2r = x2_range
    else:
        raise AssertionError("x1_range and x2_range should be either both None\
                             or non-None.")

    xx, yy = np.meshgrid(np.arange(x1r[0], x1r[1], step),
                     np.arange(x2r[0], x2r[1], step))
    f, ax = plt.subplots()
    Z = h.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.1)

    for i in range(X.shape[0]):
        c = "blue"
        if Y[i] == 1:
            c = "red"
        ax.scatter(X[i,0], X[i,1], c=c, alpha=0.5)
    plt.title(title)
    plt.show()