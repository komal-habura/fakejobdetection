from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import re
import string
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.feature_extraction.text import CountVectorizer ,TfidfVectorizer
from sklearn.model_selection import  train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from sklearn.metrics import accuracy_score , plot_confusion_matrix,confusion_matrix,classification_report
from wordcloud import wordcloud
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import  English
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import pandas as pd
punctions=string.punctuation

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("jobdection.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    inputt=[x for x in request.form.values()]
    s=""
    for i in inputt:
        s=s+i+" "
    def spacy_tokenizer(sentense):
        mytoken=parser(sentense)
        mytokens=[word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens]
        mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations]
        return mytokens
    class predictors(TransformerMixin):
        def transform(self,X,**transform_params):
            return [clean_text(text) for text in X]
        def fit(self,X,y=None ,**fit_params):
            return self
        def get_params(self,deep=True):
            return {}
        def clean_text(text):
            return text.strip().lower()
    cv =TfidfVectorizer(max_features=100)
    x=cv.fit_transform([s])
    df1=pd.DataFrame(x.toarray(),columns=cv.get_feature_names())
    prediction=model.predict_proba(df1)
    output='{0:.{1}f}'.format(prediction[0][1], 2)

    if output>str(0.5):
        return render_template('jobdection.html',pred='Your job is in fake.\nProbability of fake job  is {}'.format(output),bhai="kuch karna hain iska ab?")
    else:
        return render_template('jobdection.html',pred='Your job is real.\n Probability of real job is {}'.format(output),bhai="Your Forest is Safe for now")


if __name__ == '__main__':
    app.run(debug=True)
