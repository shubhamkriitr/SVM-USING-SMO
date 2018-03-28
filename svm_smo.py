#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 22:43:13 2018

@author: shubham
"""
import numpy as np
import matplotlib.pyplot as plt

# tolerance
TOL = 0.0001

def linear_kernel (x1, x2):
    return np.dot(x1,x2)

def quad_pol_kernel(x1, x2):
    c = np.dot(x1,x2) # 1 excluded
    return c*c

def get_rbf_kernel (sigma):
    c = -0.5/(sigma*sigma)

    def rbf_kernel(x1, x2):
        norm = np.linalg.norm(x1-x2)
        return np.exp(c*norm*norm)

    return rbf_kernel


class SVM:
    def __init__(self,C,kernel="linear", prediction_threshold=0.,verbose=False):
        name = "SVM--Trains using SMO"
        self.verbose = verbose  #..for debugging
        self.dtype = np.float32
        self.prediction_threshold = prediction_threshold
        # Uses 32-bit nums with tol as the smallest val
        self.X = None
        self.Y = None
        self.C = C
        self.kernel_name = kernel
        self.kernel = None  # kernel function
        self.classes = None
        self.alpha = None  # alpha array
        self.E = None  #  error cache
        self.W = None  #  Weight for linear case
        self.b = None  # Bias
        self.eps = TOL
        self.N = None  # num of train samples
        self.set_kernel_function(self.kernel_name)

    def feed_data(self, input_array, output_array):
        """Also resets the W,b, alpha and E"""
        assert(len(input_array.shape)==2 and len(output_array.shape)==1)
        assert(input_array.shape[0] == output_array.shape[0])
        self.X = input_array
        self.Y = output_array
        self.N = self.X.shape[0]
        #self.alpha = 0.5*self.C*np.ones(shape=(self.Y.shape[0],), dtype=self.dtype)
        self.alpha = np.zeros(shape=(self.Y.shape[0],), dtype=self.dtype)
        self.E = np.zeros(shape=(self.Y.shape[0],), dtype=self.dtype)
        self.b = 0
        if self.kernel_name == "linear":
            self.W = np.zeros(shape=(self.X.shape[1],), dtype=self.dtype)
            for i in range(self.N):
                self.W = self.W + self.Y[i]*self.alpha[i]*self.X[i]
        self._update_error_cache()

    def _update_error_cache(self):
        # TODO_ can be optimized usnig vectorized opn
        if self.verbose:
            print("Updating Error Cache...")

        for i in range(self.N):
            ui = 0
            for j in range(self.N):
                ui += self.Y[j]*self.alpha[j]*self.kernel(self.X[j],self.X[i])
            ui -= self.b
            self.E[i] = ui - self.Y[i]



    def _take_step(self, i1, i2):
        if (i1 == i2):
            return 0
        C = self.C
        alpha1_old = self.alpha[i1]
        alpha2_old = self.alpha[i2]
        y1 = self.Y[i1]
        y2 = self.Y[i2]
        E1 = self.E[i1]
        E2 = self.E[i2]
        s = y1*y2

        #  compute L and H
        if s == -1:
            L = max(0, alpha2_old - alpha1_old)
            H = min(C, C + alpha2_old - alpha1_old)
        else:
            L = max(0, alpha2_old + alpha1_old - C)
            H = min(C, alpha2_old + alpha1_old)

        if self._eql(L, H):#L == H:
            return 0

        k11 = self.kernel(self.X[i1],self.X[i1])
        k12 = self.kernel(self.X[i1],self.X[i2])
        k22 = self.kernel(self.X[i2],self.X[i2])
        eta = k11 + k22 - 2*k12

        if (eta>0):
            a2 = alpha2_old + y2*(E1 - E2)/eta
            if a2<L:
                a2 = L
            elif a2>H:
                a2 = H
        else:

            Lobj =  self._compute_cost(alpha2_old,L,E1,E2,y2,eta)
            Hobj =  self._compute_cost(alpha2_old,H,E1,E2,y2,eta)
            if (Lobj < Hobj - self.eps):
                a2 = L
            elif (Lobj > Hobj + self.eps):
                a2 = H
            else:
                a2 = alpha2_old

        if a2>H:
            a2 = H
        elif a2<L:
            a2 = L


        if (abs(a2 - alpha2_old) < self.eps*(a2 + alpha2_old + self.eps)):
            return 0

        a1 = alpha1_old + s*(alpha2_old - a2)

        if (a1 < 0):
            a2 += s*a1 # s*a1_old + a2_old - a2_new
            a1 = 0
        elif (a1 > self.C):
            t = a1 - C
            a2 += s * t
            a1 = self.C
        #Update Threshold
        b1 = (E1 + y1*(a1 - alpha1_old)*self.kernel(self.X[i1],self.X[i1])
                 + y2*(a2 - alpha2_old)*self.kernel(self.X[i1],self.X[i2])
                 + self.b)

        b2 = (E2 + y1*(a1 - alpha1_old)*self.kernel(self.X[i1],self.X[i2])
                 + y2*(a2 - alpha2_old)*self.kernel(self.X[i2],self.X[i2])
                 + self.b)

        self.b = (b1 + b2)/2.0

        #update alpha array
        self.alpha[i1] = a1
        self.alpha[i2] = a2

        #update error cache
        self._update_error_cache()

        if self.kernel_name == "linear":
            #update W
            self.W = (self.W + y1*(a1 - alpha1_old)*self.X[i1]
                             + y2*(a2 - alpha2_old)*self.X[i2])

        return 1



    def _compute_cost (alpha2_old,alpha2_new,E1,E2,y2,eta):
        return alpha2_new*((y2*(E2-E1)+eta*alpha2_old) - 0.5*eta*alpha2_new) #+ const


    def _examine_example(self, i2):
        y2 = self.Y[i2]
        alph2 = self.alpha[i2]
        E2 = self.E[i2]
        r2 = E2 * y2
        tol = self.eps
        C = self.C

        if ((r2 < (-tol)) and alph2 < C) or ((r2 > tol) and alph2 > 0):
            #skipping heuristics # TODO_ may add it later
            stop = np.random.randint(low=0,high=self.N)
            if stop == self.N -1:
                i1 = 0
            else:
                i1 = stop + 1
            while True:
                if i1 != i2 and (self._neql(self.alpha[i1],0) and
                                            self._neql(self.alpha[i1],C)):
                    if self._take_step(i1,i2):
                        return 1
                if i1 == stop:
                    break
                i1 = (i1 + 1)%self.N

            stop = np.random.randint(low=0,high=self.N)
            if stop == self.N -1:
                i1 = 0
            else:
                i1 = stop + 1

            while True:
                if i1 != i2:
                    if self._take_step(i1,i2):
                        return 1
                if i1 == stop:
                    break
                i1 = (i1 + 1)%self.N

        return 0

    def _neql(self, a, b):
        """returns true if a != b +/- tolerance"""
        if a < (b+self.eps) and a>(b-self.eps):
            return False
        return True

    def _eql(self, a, b):
        """returns true if a = b +/- tolerance"""
        if a < (b+self.eps) and a>(b-self.eps):
            return True
        return False

    def train(self, log_step=5):
        num_changed = 0
        examine_all = 1

        if self.verbose:
            print("Kernel: ",self.kernel_name,self.kernel)
            print("Training Started....")

        step_count = 0
        while (num_changed > 0 or examine_all == 1):
            num_changed = 0
            step_count+=1
            if (step_count)%log_step==0:
                Y_pred = self.predict(self.X)
                #scr, scores = self.compute_scores(self.Y, Y_pred)
                print("step={} ".format(step_count))
               # print("TPR:{} TNR:{} PPV:{} NPV:{}".format(scores["TPR"],
                     # scores["TNR"],scores["PPV"],scores["NPV"]))


            if examine_all == 1:
                for i in range(self.N):
                    #print("E_all={} loop:".format(examine_all),i)
                    num_changed += self._examine_example(i)
            else:
                for i in range(self.N):
                    if (self._neql(self.alpha[i],0) or
                                   self._neql(self.alpha[i],self.C)):
                        num_changed += self._examine_example(i)

            if examine_all == 1:
                examine_all = 0
            elif num_changed == 0:
                examine_all = 1

        if self.verbose:
            print("Training Over.")

    def set_kernel_function (self, name="linear"):
        if name != self.kernel_name:
            print("Overiding initially specified kernel.")
            print("Changing from {} to {}".format(self.kernel_name, name))
            self.kernel_name = name

        self.kernel = self._get_kernel_function(self.kernel_name)

    def _get_kernel_function(self,name='linear',sigma=1.0):
        if name == 'linear':
            return linear_kernel
        elif name == 'quad_pol':
            return quad_pol_kernel
        elif name == 'rbf':
            return get_rbf_kernel(sigma)

    def predict(self, X):
        assert(len(X.shape)==2)
        if (X.dtype != self.dtype):
            X = X.astype(self.dtype)
        Y = np.zeros((X.shape[0],),dtype=self.dtype)
        for i in range(X.shape[0]):
            Y[i] = self._predict(X[i])
        return Y

    def _predict(self, x):
        if self.kernel_name=="linear":
            op = np.dot(self.W,x) - self.b
        else:
            op = 0
            for i in range(self.N):
                op += self.alpha[i]*self.Y[i]*self.kernel(self.X[i],x)
            op -= self.b

        if op>self.prediction_threshold:
            return 1.0
        else:
            return -1.0

    def compute_scores(self,y_true, y_pred):
        cfmat = np.zeros((3,3), dtype = self.dtype)

        for i in range(y_true.shape[0]):
            cfmat[int(round(y_true[i])) + 1, int(round(y_pred[i])) + 1]+=1

        scr = {"TP":cfmat[2,2],"FP":cfmat[0,2],"TN":cfmat[0,0],"FN":cfmat[2,0]}
        TP  = scr["TP"]
        TN  = scr["TN"]
        FP  = scr["FP"]
        FN  = scr["FN"]
        TPR = (TP/(TP+FP))*100
        TNR = (TN/(TN+FN))*100
        PPV = (TP/(TP+FP))*100
        NPV = (TN/(TN+FN))*100
        return (scr, {"TPR":TPR,"TNR":TNR,"PPV":PPV,"NPV":NPV})

if __name__ == "__main__":
    print("Training SVM Using SMO on synthetic data.")
    X = np.ones(shape=(5,3),dtype=np.float32)
    K = linear_kernel  #  get_rbf_kernel(1.0)
    X[0,0] = 5
    print(K(X[0],X[1]))
