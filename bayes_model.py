import pandas as pd
import numpy as np


class BayesModel(object):

    def __init__(self,vocabulary,thetas=None):
        self.vocab=vocabulary
        if thetas==None:
            self.theta=[[],[]]
        else:
            self.theta=thetas


    def fit_laplace(self,train_df):
        total_positives=sum(train_df['y']==1)+2
        total_negatives=sum(train_df['y']==0)+2
        for col in (train_df.columns):
            if col not in ['x','y']:
                count_word_pos=sum(train_df[col].loc[train_df['y']==1])+1
                count_word_neg=sum(train_df[col].loc[train_df['y']==0])+1
                probability_word_pos=count_word_pos/total_positives
                probability_word_neg=count_word_neg/total_negatives
                self.theta[0].append(probability_word_pos)
                self.theta[1].append(probability_word_neg)

    def fit(self,train_df):

        total_positives=sum(train_df['y']==1)
        for col in (train_df.columns):
            if col not in ['x','y']:
                count_word=sum(train_df[col].loc[train_df['y']==1])
                probability_word=count_word/total_positives
                self.theta.append(probability_word)
        
    def predict(self,test_df):

        def total_probability(row):
            theta_pos=pd.Series(self.theta[0])
            theta_neg=pd.Series(self.theta[1])
            row=pd.Series(row[1:-1])
            row.index=[x for x in range(len(theta_pos))]
            temp=pd.DataFrame()
            temp['theta_pos']=theta_pos
            temp['theta_neg']=theta_neg
            temp['row']=row
            pos_predict=np.prod(temp['theta_pos'].loc[temp['row']!=0]*temp['row'].loc[temp['row']!=0])
            neg_predict=np.prod(temp['theta_neg'].loc[temp['row']!=0]*temp
            ['row'].loc[temp['row']!=0])
            likelihood_pos=round(pos_predict/(pos_predict+neg_predict),2)

            return pos_predict, neg_predict, likelihood_pos
            
        return zip(*test_df.apply(total_probability,axis=1))

    
    