import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.stem.snowball import DutchStemmer
import joblib
import os
import warnings
import nltk
import re
import csv
import psutil

class TextClassifier:
    _text = 'Text'
    _main = 'Main'
    _middle = 'Middle'
    _sub = 'Sub'
    _lbl = 'Label'
    model = None

    def __init__(self, *args, **kwargs):
        load_from_disk = kwargs.get('model_from_disk')
        self._init_lookup()
        if load_from_disk:
            self._init_model(load_from_disk)

    def _init_lookup(self):
        nltk.download('stopwords')
        # init stemmer
        self.stemmer=DutchStemmer(ignore_stopwords=True)
        self.stop_words = set(nltk.corpus.stopwords.words('dutch'))

    def _init_model(self, file):
        self.model = joblib.load(file) 

    def pickle(self, obj, file):
        joblib.dump(obj, file) 

    def export_model(self, file):
        joblib.dump(self.model, file) 

    def preprocessor(self, text):
        text = str(text)
        text=text.lower()

        # stem words
        words=re.split("\\s+",text)
        stemmed_words=[self.stemmer.stem(word=word) for word in words]
        return ' '.join(stemmed_words)


    def load_data(self, input_file, frac=1):
        _, extension = os.path.splitext(input_file)

        if extension == '.csv':
            df = pd.read_csv(input_file, sep=None, engine='python')
        elif extension == '.xlsx':
            df = pd.read_excel(input_file)
        else:
            raise Exception('Could not read input file. Extension should be .csv or .xlsx')

        print(df)

        df = df.dropna(
            axis=0, how='any',
            thresh=None,
            subset=[self._text, self._main, self._middle, self._sub],
            inplace=False
        )

        # cleanup dataset
        #df = df.drop_duplicates(subset=[self._text], keep='first')
        # for dev use only a subset (for speed purpose)
        #df = df.sample(frac=frac).reset_index(drop=True)
        # construct unique label
        df[self._lbl] = df[self._main] + "|" + df[self._middle] + "|" + df[self._sub]

        number_of_examples = df[self._lbl].value_counts().to_frame()

        # The example dataset is not large enough to train a good classification model
        # print(len(self.df),'rows valid')
        return df

    def make_data_sets(self, df, split=0.9, columns=['Middle', 'Sub']):

        texts = df[self._text]
        labels = df[columns].apply('|'.join, axis=1)

        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=1-split, stratify=labels)

        return texts, labels, train_texts, train_labels, test_texts, test_labels

    def fit(self, train_texts, train_labels):
        
        pipeline = Pipeline([
                ('vect', CountVectorizer(preprocessor=self.preprocessor, stop_words=self.stop_words)),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression()),
        ])

        # multiple hyperparameters, slow training, better optimization
        parameters_slow = {
                'clf__class_weight': (None,'balanced'), #"balanced",
                'clf__max_iter': (300,500),  #500,1000
                'clf__penalty': ('l1',), #'l2',
                'clf__multi_class': ('auto',),
                'clf__solver': ('liblinear',), # lbfgs
                'tfidf__norm': ('l2',), # 'l1'
                'tfidf__use_idf': (False,),
                'vect__max_df': (1.0,),
                'vect__max_features': (None,),
                'vect__ngram_range': ((1, 1),(1,2)) # (1,2)
        }
        # single hyperparameters, fast training, no optimization
        parameters_fast = {
                'clf__class_weight': (None,), #"balanced",
                'clf__max_iter': (300,),  #500,1000
                'clf__penalty': ('l1',), #'l2',
                #'clf__multi_class': ('auto',),
                'clf__solver': ('liblinear',), # lbfgs
                'tfidf__norm': ('l2',), # 'l1'
                'tfidf__use_idf': (False,),
                'vect__max_df': (1.0,),
                'vect__max_features': (None,),
                'vect__ngram_range': ((1, 1),) # (1,2)
        }

        grid_search = GridSearchCV(pipeline, parameters_slow,verbose=True,n_jobs=psutil.cpu_count(logical=False),cv=5)
        grid_search.fit(train_texts, train_labels)
        #print('Best parameters: ')
        #print(grid_search.best_params_)
        #print('Best score: ')
        #print(grid_search.best_score_)
        self.model = grid_search
        return grid_search
    
    def validate_model(self, test_texts, test_labels, dst_file, dst_csv, dst_validation=None):
        from sklearn.metrics import precision_score, recall_score, accuracy_score, plot_confusion_matrix
        import matplotlib.pyplot as plt

        test_predict = self.model.predict(test_texts)
        precision = str(round(precision_score(test_labels, test_predict, average='macro', zero_division=0),2))
        recall = str(round(recall_score(test_labels, test_predict, average='macro'),2))
        accuracy = str(round(accuracy_score(test_labels, test_predict),2))

        plt.rcParams["figure.figsize"] = (30,30)
        disp = plot_confusion_matrix(
            self.model,
            test_texts,
            test_labels,
            cmap=plt.cm.Blues,
            normalize=None,
            xticks_rotation='vertical')
        plt.savefig(dst_file)

        df2 = pd.DataFrame(disp.confusion_matrix, columns=disp.display_labels)
        df2.to_csv(dst_csv)
        if dst_validation:
            with open(dst_validation,'w') as csvfile:
                fieldnames = ['Text', 'predicted_category', 'actual_category']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
                writer.writeheader()
                for input, prediction, label in zip(test_texts, test_predict, test_labels):
                    if prediction != label:
                        writer.writerow(
                            {'Text': re.sub("\\W"," ",input), 'predicted_category': prediction, 'actual_category': label}
                        )
        return test_predict, precision, recall, accuracy
