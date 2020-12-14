import pandas as pd
import  numpy as np
import sklearn
import preprocessor
import model_analyzer
import bayes_model as bm
import svm_model as sm
import logistic_model as lm
import matplotlib.pyplot as plt

def k_fold(train,k, model_type='bayes',output_path=os.path.normpath(os.getcwd()+"\\results\\"),output_prefix='output',vocab_type='normal',metric='tf',analysis=True):
    train_df=pd.read_csv(train,names=['x','y'])
    scores=[]
    folds=[]
    for i in range(k):
        temp=[int((len(train_df)/k))*i,int((len(train_df)/k))*(i+1)]
        folds.append(temp)
    train_original=train_df

    for k in folds:
        print("FITTING MODEL FOR SAMPLE: "+str(k))
        temp=train_original.copy()
        test_df=temp.iloc[k[0]:k[1]]
        
        train_df=temp.drop(train_df.index[k[0]:k[1]])
        train_df.reset_index(inplace=True)
        test_df.reset_index(inplace=True)
        


        preprocessor=preprocessor.Preprocessor(train_df)

        preprocessor.preprocess('x')


        if vocab_type=='normal':
            preprocessor.create_vocabulary('x',15)
        elif vocab_type=='global':
            preprocessor.create_vocabulary2('x')
        elif vocab_type=='categorical':
            preprocessor.create_vocabulary3('x')

        if metric=='tf':
            preprocessor.update_dataframe('x','y')
        elif metric=='tfidf':
            preprocessor.update_dataframe_tfidf('x','y')


        if model_type=='bayes':
            model=bm.BayesModel(preprocessor.vocabulary)
        elif model_type=='logistic':
            model=lm.LogisticModel(preprocessor.vocabulary)
        elif model_type=='svm':
            model=sm.SVM(preprocessor.vocabulary)
        model.fit(preprocessor.data)
        

        

        analyzer=model_analyzer.Analyzer(model,test_df)


        scores.append(1-(sum(analyzer.test_data['result']==analyzer.test_data['y'])/len(analyzer.test_data)))

    return 1-np.mean(scores)

def error_graph(train_path,test_path, steps=30, model_type='bayes',output_path=os.path.normpath(os.getcwd()+"\\results\\"),output_prefix='output',vocab_type='normal',metric='tf'):
    train_df=pd.read_csv(train_path,names=['x','y'])
    test=pd.read_csv(test_path,names=['x','y'])
    scores=[]
    n_examples=[]
    for i in range(1,steps):
        temp=train_df.copy()
        temp_df=temp.iloc[:int((len(train_df)/steps))*i]
        
        test_df=test.copy()
        n_examples.append((len(train_df)/steps)*i)
        
        preprocessor=Preprocessor(temp_df)

        preprocessor.preprocess('x')


        if vocab_type=='normal':
            preprocessor.create_vocabulary('x',15)
        elif vocab_type=='global':
            preprocessor.create_vocabulary2('x')
        elif vocab_type=='categorical':
            preprocessor.create_vocabulary3('x')

        if metric=='tf':
            preprocessor.update_dataframe('x','y')
        elif metric=='tfidf':
            preprocessor.update_dataframe_tfidf('x','y')


        if model_type=='bayes':
            model=bm.BayesModel(preprocessor.vocabulary)
        elif model_type=='logistic':
            model=lm.LogisticModel(preprocessor.vocabulary)
        elif model_type=='svm':
            model=sm.SVM(preprocessor.vocabulary)
        model.fit(preprocessor.data)
        print("FITTING MODEL SAMPLE: "+str(i))

        
        if (i > 1):
            analyzer=model_analyzer.Analyzer(model,test_df)
        else:
            analyzer=model_analyzer.Analyzer(model,test_df,prep=True)
        scores.append(((sum(analyzer.test_data['result']==analyzer.test_data['y'])/len(analyzer.test_data))))
    np.savetxt(model_type+'_accuracy_nums.txt',np.array([n_examples,scores]))
    plt.plot(n_examples,scores)
    plt.ylim([0,1])
    plt.savefig(model_type+'_accuracy_graph.png')
    
def create_kfold_graph(valid_path):
    models=['bayes','logistic','svm']
    metrics=['tf','tfidf']
    labels=[]
    tf_scores=[]
    tfidf_scores=[]
    tf_scores_global=[]
    tfidf_scores_global=[]
    tf_scores_categorical=[]
    tfidf_scores_categorical=[]

    for model in models:
        for metric in metrics:
            print("********PERFORMING KFOLD ON: "+str(model)+'-'+metric)
            if metric=='tf':
                tf_scores.append(k_fold(valid_path,10,model_type=model,metric=metric,vocab_type='normal'))
                tf_scores_global.append(k_fold(valid_path,10,model_type=model,metric=metric,vocab_type='global'))
                tf_scores_categorical.append(k_fold(valid_path,10,model_type=model,metric=metric,vocab_type='categorical'))
            elif metric=='tfidf':
                tfidf_scores.append(k_fold(valid_path,10,model_type=model,metric=metric,vocab_type='normal'))
                tfidf_scores_global.append(k_fold(valid_path,10,model_type=model,metric=metric,vocab_type='global'))
                tfidf_scores_categorical.append(k_fold(valid_path,10,model_type=model,metric=metric,vocab_type='categorical'))
            labels.append(model+'-'+metric)
    bar_width=0.35
    y_pos=np.arange(len(models))
    #np.savetxt(vocab_type+'_vals.txt',np.array([tf_scores,tfidf_scores,tf_scores_global,tfidf_scores_global]))
    fig, (ax1,ax2,ax3)=plt.subplots(1,3,sharey=True)
    ax1.set_ylim([0.5,1])
    ax1.bar(y_pos-bar_width/2,tf_scores,width=bar_width,label='TF')
    ax1.bar(y_pos+bar_width/2,tfidf_scores,width=bar_width,label='TFIDF')
    ax1.set_xticks(y_pos)
    ax1.set_xticklabels(models)
    ax1.set_title('Local Vocabulary')
    
    ax2.set_ylim([0.5,1])
    ax2.bar(y_pos-bar_width/2,tf_scores_global,width=bar_width,label='TF')
    ax2.bar(y_pos+bar_width/2,tfidf_scores_global,width=bar_width,label='TFIDF')
    ax2.set_xticks(y_pos)
    ax2.set_xticklabels(models)
    ax2.set_title('Global Vocabulary')

    ax3.set_ylim([0.5,1])
    ax3.bar(y_pos-bar_width/2,tf_scores_categorical,width=bar_width,label='TF')
    ax3.bar(y_pos+bar_width/2,tfidf_scores_categorical,width=bar_width,label='TFIDF')
    ax3.set_xticks(y_pos)
    ax3.set_xticklabels(models)
    ax3.set_title('Categorical Vocabulary')
    #fig.xticks(y_pos,models)
    fig.text(0.5, 0.04, 'Models', ha='center',fontsize=14)

    fig.text(0.04, 0.5, 'Accuracy', va='center', rotation='vertical',fontsize=14)


    plt.legend()
    plt.savefig('_vocab_kfold_validation.png')

def create_bias_variance_graph(train_path,test_path, steps=30, model_type='bayes',output_path=os.path.normpath(os.getcwd()+"\\results\\"),output_prefix='output',vocab_type='normal',metric='tf'):
    train_df=pd.read_csv(train_path,names=['x','y'])
    test=pd.read_csv(test_path,names=['x','y'])
    test_scores=[]
    self_scores=[]
    n_examples=[]
    for i in range(1,steps):
        temp=train_df.copy()
        temp_df=temp.iloc[:int((len(train_df)/steps))*i]
        
        test_df=test.copy()
        n_examples.append((len(train_df)/steps)*i)
        
        preprocessor=Preprocessor(temp_df)

        preprocessor.preprocess('x')


        if vocab_type=='normal':
            preprocessor.create_vocabulary('x',15)
        elif vocab_type=='global':
            preprocessor.create_vocabulary2('x')
        elif vocab_type=='categorical':
            preprocessor.create_vocabulary3('x')
        if metric=='tf':
            preprocessor.update_dataframe('x','y')
        elif metric=='tfidf':
            preprocessor.update_dataframe_tfidf('x','y')


        if model_type=='bayes':
            model=bm.BayesModel(preprocessor.vocabulary)
        elif model_type=='logistic':
            model=lm.LogisticModel(preprocessor.vocabulary)
        elif model_type=='svm':
            model=sm.SVM(preprocessor.vocabulary)
        model.fit(preprocessor.data)
        print("FITTING MODEL SAMPLE: "+str(i))

        
        if (i > 1):
            test_analyzer=model_analyzer.Analyzer(model,test_df)
        else:
            test_analyzer=model_analyzer.Analyzer(model,test_df,prep=True)

        self_analyzer=model_analyzer.Analyzer(model,temp_df,prep=True)
        self_scores.append(((sum(self_analyzer.test_data['result']==self_analyzer.test_data['y'])/len(self_analyzer.test_data))))
        test_scores.append(((sum(test_analyzer.test_data['result']==test_analyzer.test_data['y'])/len(test_analyzer.test_data))))
    np.savetxt(model_type+'_'+vocab_type+'_'+metric+'_accuracy_nums.txt',np.array([n_examples,test_scores,self_scores]))
    test_trend=np.polyfit(n_examples,test_scores,1)
    self_trend=np.polyfit(n_examples,self_scores,1)

    p=np.poly1d(test_trend)
    q=np.poly1d(self_trend)
    plt.plot(n_examples,test_scores,'r',label="train")
    plt.plot(n_examples,self_scores,'b',label="dev")
    plt.plot(n_examples,p(n_examples),'r--')
    plt.plot(n_examples,q(n_examples),'b--')
    plt.legend()
    plt.ylim([0.5,1])
    plt.savefig(model_type+'_'+vocab_type+'_'+metric+'_bias_variance_graph.png')   