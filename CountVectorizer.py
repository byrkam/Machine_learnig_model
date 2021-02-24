import numpy as np 
import pandas as pd
import nltk as nl
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from nltk.stem import WordNetLemmatizer 

#Dataset Import
df = pd.read_csv('C:\\Users\\ans_PC\\Desktop\\DT-Datasets-part3\\MoviesDataset.csv')
#df = pd.read_csv('MoviesDataset1.csv')
summary = df['Summary'].values
y = df['Sentiment'].values

#Lemmatizer
#def clean_text(text):
#    stemmer = nl.WordNetLemmatizer()
#
#    text = nl.word_tokenize(text)
#    lemmatized_output = ' '.join([stemmer.lemmatize(w) for w in text])
#    return lemmatized_output
#
#
#for x in summary:
#    np.where(summary == x, clean_text(x), summary)
    
#Dataset Split
X_train, X_test, y_train, y_test = train_test_split(summary, y , test_size=0.2, random_state=0)

#Vectorization
vectorizer = CountVectorizer()
vectorizer.fit(X_train)

X_train = vectorizer.transform(X_train)
X_test  = vectorizer.transform(X_test)

#print(X_train)

#Model
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

#Accuracy
accuracy = classifier.score(X_test, y_test)
print('\n Accuracy for data: ' ,accuracy , "\n")

y_pred = classifier.predict(X_test)
C = confusion_matrix(y_test,y_pred)
print('Confussion Matrix : \n', C , "\n" )

  
C_percentage = C / C.astype(float).sum(axis=1)
print("Confussion Matrix % : \n" , C_percentage )


