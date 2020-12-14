import pandas as pd
import numpy as np
import model_analyzer 
import logistic_model as lm
import bayes_model as bm
import matplotlib.pyplot as plt
import nltk
from nltk import TextCollection
from nltk.tokenize import RegexpTokenizer
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from nltk import FreqDist
import string
from sklearn.metrics import confusion_matrix
import os
from time import time
import svm_model as sm
import warnings
warnings.filterwarnings("ignore")



#purpose of this module is to make it straightforward and simple to conduct preprocessing for NLP projects
class Preprocessor(object):

    #initialize preprocessor object by passing in a Pandas dataframe
    def __init__(self,dataframe,vocabulary=None):
        self.data=dataframe
        self.vocabulary=vocabulary

    #specify column title to be tokenized
    #tokenization refers to the 'splitting up' of individual words within a block of text
    def tokenize(self,column):
        tokenizer = RegexpTokenizer('\w+')
        self.data[column]=(self.data[column].apply(tokenizer.tokenize))

    
    
    #lemmatization refers to the reformatting of tokenized words
    #ie. removing 'stop words', punctuation, numbers etc.
    def lemmatize(self,column):
        stop_words=stopwords.words('english')
        stop_words.append('br')
        def lemmatize_tokens(tokens):
            lemmatizer=WordNetLemmatizer()
            output=[]

            for word, tag in tokens:
                word = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                            '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', word)
                word = re.sub("(@[A-Za-z0-9_]+)","", word)
                word = re.sub("[0-9]+","",word)
                if tag.startswith('NN'):
                    pos='n'
                elif tag.startswith('VB'):
                    pos='v'
                else:
                    pos='a'
                if word.lower() not in stop_words and word not in string.punctuation:
                    output.append(lemmatizer.lemmatize(word.lower(),pos))
            return output
        
        self.data[column]=self.data[column].apply(lemmatize_tokens)

    def lemmatize_tokens(self,tokens):
        stop_words=stopwords.words('english')
        stop_words.append('br')
        lemmatizer=WordNetLemmatizer()
        output=[]


        for word, tag in tokens:
            word = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                        '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', word)
            word = re.sub("(@[A-Za-z0-9_]+)","", word)
            word = re.sub("[0-9]+","",word)
            if tag.startswith('NN'):
                pos='n'
            elif tag.startswith('VB'):
                pos='v'
            else:
                pos='a'
            if word.lower() not in stop_words and word not in string.punctuation:
                output.append(lemmatizer.lemmatize(word.lower(),pos))

        return output


    def preprocess(self,column):
        tokenizer = RegexpTokenizer('\w+')
        acc=[]
        count=0

        for text in self.data[column]:
            
            x=tokenizer.tokenize(text)
            x=list(pos_tag(x))
            
            x=self.lemmatize_tokens(x)
            acc.append(x)
            count+=1
        acc=pd.Series(acc)

        self.data[column]=acc

    #takes an integer to determine the number of words we want to use per text in order to contribute to our overall vocabulary
    def create_vocabulary(self, column, num_words_per_text):
        vocabulary=[]
        for lemmatized_text in self.data[column]:

            freq_dist=FreqDist(lemmatized_text)
            count=0
            for word,times in freq_dist.most_common(50):
                if count==10:
                    break
                elif word not in vocabulary and times>1:
                    vocabulary.append(word)
                    count+=1
        self.vocabulary= vocabulary


    def create_vocabulary2(self,column):
        vocabulary=[]
        for text in self.data[column]:
            vocabulary= vocabulary+text
        freq_dist_vocab=FreqDist(vocabulary).most_common()

        vocab_acc=[]
        for i in range(len(freq_dist_vocab)):
            #print(freq_dist_vocab)
            #print(freq_dist_vocab[i])
            if freq_dist_vocab[i][1]>=5:
                vocab_acc.append(freq_dist_vocab[i][0])
            else:
                break
        self.vocabulary=vocab_acc
        # self.vocabulary=[i[0] for i in freq_dist_vocab.most_common(vocab_length)]
        
    def create_vocabulary3(self,column):
        both=[y for x in (self.data[column]) for y in x]
        pos_voc=[y for x in (self.data[column].loc[self.data['y']=='positive']) for y in x]
        neg_voc=[y for x in (self.data[column].loc[self.data['y']=='negative']) for y in x]
        
        freq_dist_both=FreqDist(both).most_common()
        freq_dist_pos=dict(FreqDist(pos_voc).most_common())
        freq_dist_neg=dict(FreqDist(neg_voc).most_common())
        acc=[]
        for i in range(len(freq_dist_both)):
            if freq_dist_both[i][1]>5:
                try:
                    min_rep=min([freq_dist_pos[freq_dist_both[i][0]],freq_dist_neg[freq_dist_both[i][0]]])
                    max_rep=max([freq_dist_pos[freq_dist_both[i][0]],freq_dist_neg[freq_dist_both[i][0]]])
                    if min_rep<.7*max_rep:
                        acc.append(freq_dist_both[i][0])
                    if freq_dist_both[i][0]=='movie':
                        print("POS: "+str(freq_dist_pos[freq_dist_both[i][0]]))
                        print("NEG: "+str(freq_dist_neg[freq_dist_both[i][0]]))
                except KeyError:
                    pass
        if acc ==[]:
            print("ERROR")
            exit()
        self.vocabulary=acc

    #once a vocabulary has been established, this function recreates the dataframe
    #it creates a specific column for each word in the vocabulary and includes the number of times that word
    #is found in each text
    def update_dataframe(self,text_column,y_column):
        new_df=pd.DataFrame()
        new_df['x']=self.data[text_column]
        
        for j in range(len(self.vocabulary)):
            word=self.vocabulary[j]

            new_df['x'+str(j+1)]=new_df['x'].apply((lambda x: x.count(word)))
        def make_binary(entry):
            if entry=='positive':
                return 1
            else:
                return 0
        new_df['y']=self.data[y_column].apply(make_binary)
        self.data=new_df


    def get_TFIDF(self,word,word_count,text,collection_of_texts,number_docs_with_word):
    
        TF=word_count/len(text)
        total_docs=len(collection_of_texts)

        IDF=np.log(total_docs/number_docs_with_word)
        TFIDF=TF*IDF
        return TFIDF
    def update_dataframe_tfidf(self,text_column,y_column):
        new_df=pd.DataFrame()

        new_df['x']=self.data[text_column]
        def docs_with_word(word,collection_of_texts):
            result=0
            for text in collection_of_texts:
                if word in text:
                    result+=1
            if result==0:
                return 1
            else:
                return result
        
        for j in range(len(self.vocabulary)):
            word=self.vocabulary[j]
            number_docs_with_word=docs_with_word(word,new_df['x'])

            new_df['x'+str(j+1)]=new_df['x'].apply(lambda x: self.get_TFIDF(word,x.count(word),x,new_df['x'],number_docs_with_word))*100
        def make_binary(entry):
            if entry=='positive':
                return 1
            else:
                return 0
        new_df['y']=self.data[y_column].apply(make_binary)
        self.data=new_df

def make_binary(entry):
    if entry=='positive':
        return 1
    else:
        return 0
def run_experiment(train,test,model_type='bayes',output_path=os.path.normpath(os.getcwd()+"\\results\\"),output_prefix='output',vocab_type='normal',metric='tf',analysis=True):
    train_df=pd.read_csv(train,names=['x','y'])
    test_df=pd.read_csv(test,names=['x','y'])
    t=time()
    print("----------------------------")
    print("CONDUCTING EXPERIMENT - "+model_type)
    print("Vocabulary: %s\tMetric: %s\tAnalysis: %s" %(vocab_type,metric,analysis))
    preprocessor=Preprocessor(train_df)
    preprocessor.preprocess('x')

    duration =time()-t
    
    print("Time to preprocess: %f" %(duration))
    print("CREATING VOCABULARY")
    t=time()
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
    duration =time()-t
    preprocessor.data.to_csv('preprocessed_train_data.csv')
    print("Time to create vocab and update counts: %f" %(duration))
    t=time()
    if model_type=='bayes':
        model=bm.BayesModel(preprocessor.vocabulary)
    elif model_type=='logistic':
        model=lm.LogisticModel(preprocessor.vocabulary)
    elif model_type=='svm':
        model=sm.SVM(preprocessor.vocabulary)
    print("FITTING MODEL")
    model.fit(preprocessor.data)
    duration =time()-t

    print("Time to fit model: %f" %(duration))
    print("PREDICTING")
    t=time()
    test_df=pd.read_csv(test, names=['x','y'])
    if analysis:
        
        analyzer=model_analyzer.Analyzer(model, test_df, metric=metric)
        thresholds=[round((0.05*x),2) for x in range(20,-1,-1)]
        print("ACCURACY: "+str(sum(analyzer.test_data['result']==analyzer.test_data['y'])/len(analyzer.test_data)))
        #analyzer.threshold_scan(thresholds,output_path+"\\"+output_prefix+'_threshold_scan.csv')
        #analyzer.print_prob_distribution(output_path+"\\"+output_prefix+'_prob_distribution.png')
        analyzer.print_confusion_matrix(threshold=0.5, output_path=output_path+"\\"+output_prefix+'__confusion_matrix.png')
    else:
        analyzer=model_analyzer.Analyzer(model, test_df, metric)
    try:
        model.get_word_cloud('pos')
        model.get_word_cloud('neg')
    except:
        pass
    duration =time()-t
    print("Time to predict and output analysis: %f" %(duration))


  

def main(train_path, valid_path, test_path):
    
    run_experiment(train_path,test_path,model_type='svm',output_prefix='test',vocab_type='categorical',analysis=True,metric='tfidf')


if __name__=='__main__':
    main('train.csv','valid.csv','test.csv')
    