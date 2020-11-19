import pandas as pd
import numpy as np


class BayesModel(object):

    def __init__(self,vocabulary,threshold=0.5,thetas=None):
        self.vocab=vocabulary
        self.threshold=threshold
        if thetas==None:
            self.theta=[[],[]]
        else:
            self.theta=thetas


    def fit(self,train_df):
        total_positives=sum(train_df['y']==1)+len(self.vocab)
        total_negatives=sum(train_df['y']==0)+len(self.vocab)
        for col in (train_df.columns):
            if col not in ['x','y']:
                count_word_pos=sum(train_df[col].loc[train_df['y']==1])+1
                count_word_neg=sum(train_df[col].loc[train_df['y']==0])+1
                probability_word_pos=count_word_pos/total_positives
                probability_word_neg=count_word_neg/total_negatives
                self.theta[0].append(probability_word_pos)
                self.theta[1].append(probability_word_neg)

        

    def predict(self,test_df):

        def predict_row(row):

            
            this_row=row.iloc[1:-1]
            this_row.index=[x for x in range(len(this_row))]
            pos= np.prod(np.multiply(this_row.loc[this_row>0],pd.Series(self.theta[0]).iloc[this_row[this_row>0].index]))
            neg=np.prod(np.multiply(this_row.loc[this_row>0],pd.Series(self.theta[1]).iloc[this_row[this_row>0].index]))

            if pos==0:
                return 0
            if (pos/(pos+neg))>self.threshold:
                return 1
            else:
                return 0

        return test_df.apply(predict_row,axis=1)



    
    