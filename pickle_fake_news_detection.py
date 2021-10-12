import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.linear_model import PassiveAggressiveClassifier

news = pd.read_csv('FN_train.csv')
conversion_dict = {0: 'Real', 1: 'Fake'}
news['label'] = news['label'].replace(conversion_dict)
X = news['text']
y = news['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),
                    ('PAC', PassiveAggressiveClassifier())])

pipeline.fit(X_train.values.astype('U'), y_train)
pred = pipeline.predict(X_test.values.astype('U'))

print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))

with open('model.pickle', 'wb') as handle:
    pickle.dump(pipeline, handle, protocol=pickle.HIGHEST_PROTOCOL)
