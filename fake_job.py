#!/usr/bin/env python
# coding: utf-8

# In[4]:


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


# In[5]:


import pandas as pd
import seaborn as sns


# In[6]:


df=pd.read_csv("fake_job_postings.csv")


# In[7]:


df.head()


# In[8]:


df.isnull().sum()


# In[9]:


col=["job_id","telecommuting","has_company_logo","has_questions","salary_range","employment_type"]


# In[10]:


for i in col:
    del(df[i])


# In[11]:


df.head()


# In[12]:


df.fillna("",inplace=True)


# In[13]:


df.head()


# In[14]:


plt.figure(figsize=(15,5))
sns.countplot(y="fraudulent",data=df)


# In[15]:


df.groupby("fraudulent")["fraudulent"].count()


# In[16]:


exp=dict(df.required_experience.value_counts())


# In[17]:


del(exp[""])


# In[18]:


exp


# In[19]:


plt.figure(figsize=(10,5))
sns.set_style(style="whitegrid")
plt.bar(exp.keys(),exp.values())
plt.title("No of job with experience",size=20)
plt.xlabel("Experience",size=10)
plt.ylabel("No of jobs",size=10)
plt.show()


# In[20]:


def split(location):
    l=location.split(",")
    return l[0]
df["country"]=df.location.apply(split)


# In[21]:


df.head()


# In[22]:


country=dict(df.country.value_counts()[:14])
del(country[""])
country


# In[23]:


plt.figure(figsize=(15,5))
plt.title("country-wise job posting",size=20)
plt.bar(country.keys(),country.values())
plt.ylabel("no of jobs",size=15)
plt.xlabel("countries",size=15)


# In[24]:


education=dict(df.required_education.value_counts()[:7])
del education[""]


# In[25]:


education


# In[26]:


plt.figure(figsize=(10,5))
plt.bar(education.keys(),education.values())


# In[27]:


print(df[df.fraudulent==0].title.value_counts()[:10])


# In[28]:


print(df[df.fraudulent==1].title.value_counts()[:10])


# In[29]:


df["text"]=df["title"]+" "+df["company_profile"]+" "+df["description"]+" "+df["requirements"]+" "+df["benefits"]
del df["title"]
del df["location"]
del df["department"]
del df["company_profile"]
del df["description"]
del df["requirements"]
del df["benefits"]
del df["required_experience"]
del df["required_education"]
del df["industry"]
del df["function"]
del df["country"]


# In[30]:


df.head()


# In[31]:


fraudjobs_text = df[df.fraudulent==1].text
realjobs_text = df[df.fraudulent==0].text


# In[32]:


from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# In[33]:


plt.figure(figsize=(16,14))
wc = WordCloud(min_font_size =3 , max_words = 3000,width=1600,height=800,stopwords=STOPWORDS).generate(str(" ".join(fraudjobs_text)))
plt.imshow(wc,interpolation='bilinear')


# In[34]:


STOPWORDS= spacy.lang.en.stop_words.STOP_WORDS
plt.figure(figsize=(16,14))
wc = WordCloud(min_font_size =3 , max_words = 3000,width=1600,height=800,stopwords=STOPWORDS).generate(str(" ".join(realjobs_text)))
plt.imshow(wc,interpolation='bilinear')


# In[35]:


import spacy
punctions=string.punctuation


# In[36]:


stop_word=spacy.lang.en.stop_words.STOP_WORDS
parser=English()


# In[37]:


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


# In[38]:


cv =TfidfVectorizer(max_features=100)
x=cv.fit_transform(df["text"])
df1=pd.DataFrame(x.toarray(),columns=cv.get_feature_names())
df.drop(["text"],axis=1,inplace=True)
main_df = pd.concat([df1,df],axis=1)


# In[39]:


main_df.head()


# In[40]:


X=main_df.iloc[:,:-1]
Y=main_df.iloc[:,-1]


# In[41]:


x_train,x_test,y_train,y_test= train_test_split(X,Y,test_size=0.30)


# In[42]:


x_test.shape


# In[43]:


from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
clf = MLPClassifier().fit(x_train, y_train)


# In[50]:


k=x_test.iloc[:1,:]


# In[52]:


p=clf.predict(k)


# In[56]:


clf.score(x_test, y_test)


# In[54]:


import pickle


# In[58]:


pickle.dump(clf,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))








