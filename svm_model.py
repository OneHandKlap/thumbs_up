import pandas as pd
import numpy as np
from sklearn import svm
class SVM(object):
    def __init__(self,vocabulary,thetas=None,threshold=0.5):
        self.vocab=vocabulary
        if thetas==None:
            self.theta=np.array([1]*len(vocabulary))
        self.verbose=True
        self.SVM=None
        self.threshold=threshold
    def fit(self, train_df):
        xs=train_df.iloc[:,1:-1]
        y=train_df['y']
        self.SVM=svm.SVC(kernel='rbf')
        self.SVM.fit(xs,y)

    def predict(self,test_df):
        xs=test_df.iloc[:,1:-1]
        
        return self.SVM.predict(xs)