# coding: utf-8

import pandas as pd
import requests
import numpy as np
import re
import pickle
import multiprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.externals import joblib
from sklearn.metrics import classification_report

########################################################################################
# data path
path = '/home/maarten/Documents/projecten/SIA/data/'
cpu_count = multiprocessing.cpu_count()

########################################################################################

def load_sia():
    '''
    Load data from SIA daily dump
    '''
    
    # load .csv's
    df = pd.read_csv(path+'signals.csv')
    categories = pd.read_csv(path+'categories.csv')

    # merge
    df = pd.read_csv(path+'signals.csv')
    categories = categories.drop_duplicates('_signal_id',keep='last').set_index('_signal_id')
    cat_main = categories['main'].to_dict()
    cat_sub = categories['sub'].to_dict()
    df['main'] = df['id'].map(lambda x: cat_main[x])
    df['sub'] = df['id'].map(lambda x: cat_sub[x])
        
    return df.drop_duplicates('id',keep='last')[['main','sub','text']]

########################################################################################

def load_mora():
    '''
    Load data from old system
    '''
    filename_mora = path + 'MORA-2014-2018-28mei.csv'
    mora = pd.read_csv(filename_mora)[['Hoofdrubriek','Subrubriek','Toelichting','Datum melding']]
    mora['Datum melding'] =pd.to_datetime(mora['Datum melding'])
    mora = mora[mora['Datum melding']>'2017-01-01']
    mora['main'] = mora['Hoofdrubriek']
    mora['sub'] = mora['Subrubriek']
    mora['text'] = mora['Toelichting']
    mora = mora.reset_index(drop=True)[['main','sub','text']]
    return mora

########################################################################################

def load_waternet():
    '''
    Load data from waternet
    '''
    filename_waternet = path + 'datadump2014-2017.csv'
    waternet = pd.read_csv(filename_waternet ,encoding='latin', sep=',')
    waternet['main'] = 'Overlast op het water'
    waternet = waternet[waternet['Soort melding'] != 'Handhaving']
    def go(x):
        try:
            return x#.split('-')[1]
        except:
            return x
    waternet['sub'] = waternet['Soort melding'].map(lambda x: go(x))
    waternet['text'] = waternet['Opmerkingen']

    waternet = waternet[['main','sub','text']].dropna()
    return waternet

########################################################################################

def import_klok():
    '''
    Load data from csv
    '''
    df = pd.read_csv(path + 'klokken_v2.csv')
    df['text']  =df['text']
    df['sub'] = 'Klok'
    df['main'] = 'Wegen, verkeer, straatmeubilair'
    return df[['text','sub','main']]


########################################################################################

def create_model(train_texts, test_texts, train_labels, test_labels):
    '''
    Create a model and seek optimal parameters using Gridsearch cross validation
    '''

    # pipeline of classifier
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', LogisticRegression()),
    ])



    # multiple hyperparameters, slow training, better optimization
    parameters = {
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
    parameters = {
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

    grid_search = GridSearchCV(pipeline, parameters,verbose=True,n_jobs=cpu_count-1,cv=5)
    grid_search.fit(train_texts, train_labels)

    print('TF-IDF + Log')
    print('Best parameters: ')
    print(grid_search.best_params_)
    print('')

    print('Best score gridsearch: ')
    print(grid_search.best_score_)
    print('')
    return grid_search

########################################################################################

def print_eval(model_name):
    '''
    Returns evaluation of the model
    '''

    model = joblib.load(model_name)

    predict = model.predict(test_texts)

    rep = classification_report(test_labels,predict, output_dict=True)

    precision = precision_score(test_labels,predict, average='macro')
    recall = recall_score(test_labels,predict, average='macro')
    f1 = f1_score(test_labels,predict, average='macro')
    accuracy = accuracy_score(test_labels,predict)

    return {'accuracy':accuracy,'precision':precision,'recall':recall,'f1':f1, 'rep':rep}

########################################################################################
# loading active classes from SIA api

r = requests.get('https://acc.api.data.amsterdam.nl/signals/v1/public/terms/categories/')
cats = r.json()
cats = pd.DataFrame(cats['results'])
main_dict = cats.set_index('name')['slug'].to_dict()

########################################################################################
# loading all sources

df = pd.concat([load_sia(),load_waternet(),load_mora(),import_klok()], sort=True).reset_index(drop=True)

print(len(df),'rows of trainning and testing data loaded')

# collecting active categories
b = []
for a in list(cats['sub_categories']):
    b+=a
sub_cats = pd.DataFrame(b)
sub_dict = sub_cats.set_index('name')['slug'].to_dict()

sub_cats = sub_cats[sub_cats['is_active']]
sub_cats['slug'] = sub_cats['_links'].map(lambda x: x['self']['href'])

df['is_active'] = df['sub'].isin(sub_cats['name'])
df['is_overig'] = df['sub'].map(lambda x: 'overig' in str(x).lower())

number_of_examples = df[df['is_active']==True][df['is_overig']==False]['sub'].value_counts().to_frame()
df['is_bigger_than_50'] = df['sub'].isin(number_of_examples[number_of_examples['sub']>50].index)

print(len(list(set(df[df['is_active']==True]['sub']))),'sub classes are active')
print(len(list(set(df[df['is_active']==True][df['is_overig']==False]['sub']))),'sub classes are active and not overig')
print(len(list(set(df[df['is_active']==True][df['is_overig']==False][df['is_bigger_than_50']==True]['sub']))),'sub classes are active and not overig and bigger than 50')

df = df[df['is_active']==True][df['is_overig']==False][df['is_bigger_than_50']==True]
df['sub_slug'] = df['sub'].map(lambda x: sub_dict[x])
df['main_slug'] = df['main'].map(lambda x: main_dict[x])

########################################################################################
# create features and classes

texts = list(df['text'])



texts = [ str(x) for x in texts]
labels = list(df['main_slug'])

print('texts ',len(texts))
print('main classes',len(list(set(labels))))
for l in list(set(labels)):
    print(l)

# create train/test split
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size = 0.2, random_state = 1337,shuffle=True)

# train main model
main_model = create_model(train_texts, test_texts, train_labels, test_labels)

# dump main model
joblib.dump(main_model, 'main_model.pkl') 

# eval main model
main_eval = print_eval('main_model.pkl') 
print(main_eval)

# dump slugs
main_slugs = ['https://api.data.amsterdam.nl/signals/v1/public/terms/categories/'+x for x in main_model.classes_]
pickle.dump( main_slugs, open( "main_slugs.pkl", "wb" ) )

########################################################################################
# randomize dataset

df = df.sample(frac=1).reset_index(drop=True)

# create features and classes
texts = list(df['text'])
texts = [ str(x) for x in texts]
labels = list(df['sub_slug'])

print('sub classes',len(list(set(labels))))
for l in list(set(labels)):
    print(l)

# create train/test split
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size = 0.2, random_state = 1337,shuffle=True)

# train sub model
sub_model = create_model(train_texts, test_texts, train_labels, test_labels)

# dump sub model
joblib.dump(sub_model, 'sub_model.pkl') 

# eval sub mode
sub_eval = print_eval('sub_model.pkl') 
print(sub_eval)

########################################################################################
# dump class slugs to pkl

df['both_slugs'] = 'https://api.data.amsterdam.nl/signals/v1/public/terms/categories/'+df['main_slug']+'/sub_categories/'+df['sub_slug']

sub_slugs = []
for s in sub_model.classes_:
    for a in list(set(df['both_slugs'])):
        if s  == a.split('sub_categories/')[1]:
            sub_slugs.append(a)
            break

pickle.dump( sub_slugs, open( "sub_slugs.pkl", "wb" ) )

########################################################################################