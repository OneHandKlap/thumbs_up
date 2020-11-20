import pandas as pd
import numpy as np
import model_analyzer
from bayes_model import BayesModel
from logistic_regression import LogisticRegression
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
import model_analyzer


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

    #this function adds identifiers to individual words which indicates their 'part of speech'
    def add_tags(self,column):
        self.data[column]=self.data[column].apply(pos_tag)

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

    ##TODO CREATE NEW VOCAB FUNCTION##

    def create_vocabulary2(self,column,vocab_length):
        vocabulary=[]
        for text in self.data[column]:
            vocabulary= vocabulary+text
        freq_dist_vocab=FreqDist(vocabulary)


        self.vocabulary=[i[0] for i in freq_dist_vocab.most_common(vocab_length)]
        
    #returns the n words with highest TFIDF scores
    def tfidf(self,n):
        collection_of_texts=self.data['x']
        freq_dist_collection=[]
        total_docs=len(collection_of_texts)
        def docs_with_word(word,collection_of_texts):
            result=0
            for text in collection_of_texts:
                if word in text:
                    result+=1
            if result==0:
                return 1
            else:
                return result
        for text in collection_of_texts:
            freq_dist_collection.append(FreqDist(text))
        tdif_acc={}
        for text in collection_of_texts:
            freq_dist_text=FreqDist(text)
            
            for word in freq_dist_text.most_common(1000):
                TF=word[1]/len(text)

                number_docs_with_word=docs_with_word(word[1],collection_of_texts)

                IDF=np.log(total_docs/number_docs_with_word)

                tdif_acc[word[0]]=TF*IDF
        tdif_acc=sorted(tdif_acc.items(),key=lambda x:x[1],reverse=True)
        print(tdif_acc)
        tdif_acc=[x[0] for x in tdif_acc]
        

        self.vocabulary= tdif_acc[:n]


    #once a vocabulary has been established, this function recreates the dataframe
    #it creates a specific column for each word in the vocabulary and includes the number of times that word
    #is found in each text
    def update_dataframe(self,text_column,y_column):
        new_df=pd.DataFrame()
        new_df['x']=self.data[text_column]
        
        for j in range(len(self.vocabulary)):
            new_df['x'+str(j+1)]=[None]*len(new_df)
            for i in range(len(new_df['x'])):
                
                count=0
                for word in new_df['x'][i]:

                    if word==self.vocabulary[j]:
                        count+=1

                new_df['x'+str(j+1)][i]=count
        def make_binary(entry):
            if entry=='positive':
                return 1
            else:
                return 0
        new_df['y']=self.data[y_column].apply(make_binary)
        self.data=new_df
        #new_df.to_csv('output.csv')

    def plot_confusion(self):
        print(self.data['y'])
        print(self.data['like'])

def make_binary(entry):
    if entry=='positive':
        return 1
    else:
        return 0
def run_experiment(train,test,output_path='C:\\Users\\pabou\\Documents\\GitHub\\CPS803-Machine_Learning\\thumbs_up\\results\\',output_prefix='output',vocab_type='normal',analysis=True):
    train_df=pd.read_csv(train,names=['x','y'])
    preprocessor=Preprocessor(train_df)
    preprocessor.tokenize('x')
    preprocessor.add_tags('x')
    preprocessor.lemmatize('x')
    if vocab_type=='normal':
        preprocessor.create_vocabulary('x',15)
    elif vocab_type=='global':
        preprocessor.create_vocabulary2('x',len(train_df)*5)
    elif vocab_type=='tfidf':
        preprocessor.tfidf(len(train_df)*5)
    preprocessor.update_dataframe('x','y')

    model=BayesModel(preprocessor.vocabulary)
    model.fit_laplace(preprocessor.data)

    if analysis:
        analyzer=model_analyzer.Analyzer(model, test)
        thresholds=[round((0.05*x),2) for x in range(20,-1,-1)]
        analyzer.threshold_scan(thresholds,output_path+output_prefix+'_threshold_scan.csv')
        analyzer.print_prob_distribution(output_path+output_prefix+'_prob_distribution.png')
        analyzer.print_confusion_matrix(threshold=0.5, output_path=output_path+output_prefix+'__confusion_matrix.png')

def main(train_path,test_path):
    print("RUNNING NORMAL")
    run_experiment(train_path,test_path,output_prefix='normal')
    print("RUNNING GLOBAL")
    run_experiment(train_path,test_path,output_prefix='global',vocab_type='global')
    run_experiment(train_path,test_path,output_prefix='tfidf',vocab_type='tfidf')

    


    

    

if __name__=='__main__':
    main('train_small.csv','test.csv')
    