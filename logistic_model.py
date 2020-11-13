import pandas as pd
import numpy as np

class LogisticModel(object):

    def __init__(self,vocabulary,alpha=0.00005,max_iter=1000,thetas=None):
        self.vocab=vocabulary
        if thetas==None:
            self.theta=np.array([1]*len(vocabulary))

        self.alpha=alpha
        self.max_iterations=max_iter

    def fit_batch_GA(self,train_df):
        # y=np.array(train_df['y'])
        # xs=np.array(train_df.iloc[:,1:-1])
        # delta = self.alpha*np.dot(np.exp(np.dot(self.theta.T,xs.T)),xs)
        # self.theta=np.add.reduce([self.theta,delta])
        theta_acc=[]
        for count in range(10):
            print(count)
            for i in range(len(train_df)):
                this_row_xs=np.array(train_df.iloc[i].iloc[1:-1])
                prediction=np.matmul(self.theta[this_row_xs.nonzero()],this_row_xs
                [this_row_xs.nonzero()])
                sigmoid_prediction=1/(1+np.exp(-prediction))
                delta=(train_df['y'][i]-sigmoid_prediction)
                delta2=delta*this_row_xs[this_row_xs.nonzero()]
                
                self.theta[this_row_xs.nonzero()]=self.theta[this_row_xs.nonzero()]+self.alpha*delta2
        print(self.theta)
        

    def predict(self,test_df):
        print(self.theta)
        pred=[]

        for i in range(len(test_df)):
            xs=np.array(test_df.iloc[i].iloc[1:-1])
            
            pred.append(1/1+(np.exp(np.matmul(self.theta[xs.nonzero()],xs[xs.nonzero()]))))
        return (pred)


    