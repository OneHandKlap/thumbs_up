import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn
import preprocessor 
import bayes_model

class Analyzer(object):

    def __init__(self,model,test_path,metric='tf'):
        self.model=model
        self.test_path=test_path
        self.scan=None
        self.test_data = self.make_dataframe(metric)


    def make_dataframe(self,metric):
        test_df=pd.read_csv(self.test_path,names=['x','y'])

        test_data=preprocessor.Preprocessor(test_df,self.model.vocab)
        test_data.tokenize('x')
        test_data.add_tags('x')
        test_data.lemmatize('x')
        if metric =='tf':
            test_data.update_dataframe('x','y')
        elif metric =='tfidf':
            test_data.update_dataframe_tfidf('x','y')

        test_data.data['likelihood_pos']=(self.model.predict(test_data.data))

        test_data.data['over_50']=test_data.data['likelihood_pos']>0.5
        return test_data.data
    def threshold_scan(self,thresholds,output_path):
        metrics=pd.DataFrame()

        
        def make_judgement(row,threshold):
            
            if row['likelihood_pos']>threshold:
                return 1
            else:
                return 0
        
        metric_acc=[]
        best_confusion=[[0,100],[100,0]]
        for i in thresholds:

            self.test_data['results']=self.test_data.apply(make_judgement, args=(i,),axis=1)
            self.test_data['results']=self.test_data['results'].astype('bool')
            self.test_data['y']=self.test_data['y'].astype('bool')
            count_true=sum(self.test_data['results'])
            count_false=sum(~self.test_data['results'])
            label_true=sum(self.test_data['y'])
            label_false=sum(~self.test_data['y'])
            true_pos=sum(self.test_data['results']&self.test_data['y'])
            true_neg=sum(~self.test_data['results']&~self.test_data['y'])
            false_pos=sum(self.test_data['results']&~self.test_data['y'])
            false_neg=sum(~self.test_data['results']&self.test_data['y'])
            accuracy=(true_pos+true_neg)/(true_pos+true_neg+false_pos+false_neg)

            #determine if this is the best confusion matrix so far
            if true_pos+true_neg>(best_confusion[0][0]+best_confusion[1][1]):
                best_confusion=[[true_pos,false_pos],[false_pos,true_neg]]
            elif false_neg+false_pos<(best_confusion[0][1]+best_confusion[1][0]):
                best_confusion=[[true_pos,false_pos],[false_neg,true_neg]]
            elif false_neg+false_pos==(best_confusion[0][1]+best_confusion[1][0]):
                if false_neg<best_confusion[1][0]:
                    best_confusion=[[true_pos,false_pos],[false_neg,true_neg]]
            try:
                precision=true_pos/(true_pos+false_pos)
            except ZeroDivisionError:
                precision= 0
                
            try:
                recall=true_pos/(true_pos+false_neg)
            except ZeroDivisionError:
                recall=0
            try:
                spec=true_neg/(true_neg+false_pos)
            except ZeroDivisionError:
                spec=0
            try:
                f1=(2*precision*recall)/(precision+recall)
            except ZeroDivisionError:
                f1=0

            metric_acc.append([i,true_pos,true_neg,false_pos,false_neg,accuracy,precision,recall,spec,f1])

        metrics=pd.DataFrame(metric_acc)

        
        metrics.columns=['threshold','tp','tn','fp','fn','accuracy','precision','recall','specificity','harmonic_mean']
        metrics.to_csv(output_path+"threshold_scan.csv")
        self.threshold_scan=metrics
        return

    def print_prob_distribution(self,output_path=None):


        plt.figure()
        os=self.test_data['likelihood_pos'].loc[self.test_data['y']==1]
        xs=self.test_data['likelihood_pos'].loc[self.test_data['y']==0]
        
        plt.plot([0 for x in range(len(os))],os,'go' ,linewidth=2,label="label==1")
        plt.plot([0 for x in range(len(xs))],xs, 'bx',linewidth=2,label="label==0")

        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off 
            labelbottom=False)
        plt.ylabel("Probability")
        plt.title(output_path)
        plt.legend()
        if output_path is not None:
            plt.savefig(output_path)
        else:
            plt.show()

    def print_confusion_matrix(self,threshold=0.5,output_path=None):

        #confusion matrix plot borrowed from "https://vitalflux.com/python-draw-confusion-matrix-matplotlib/"
        
        conf_matrix=confusion_matrix(self.test_data['y'],self.test_data['likelihood_pos'].apply(lambda x: x>=threshold))
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(x=i, y=j,s=conf_matrix[j, i], va='center', ha='center', size='xx-large')
        
        plt.xlabel('Predictions', fontsize=18)
        plt.ylabel('Actuals', fontsize=18)
        plt.title(output_path, fontsize=18)
        if output_path is not None:
            plt.savefig(output_path)
        else:
            plt.show()