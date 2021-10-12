import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("FN_train.csv")
#print(df)

conversion_dict = {0: 'Real', 1: 'Fake'}
df['label'] = df['label'].replace(conversion_dict)
#print(df)
print(df.label.value_counts())

x_train,x_test,y_train,y_test=train_test_split(df['text'], df['label'], test_size=0.25, random_state=7, shuffle=True)

tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.75)

#text to vec
vec_train= tfidf_vectorizer.fit_transform(x_train.values.astype('U'))
vec_test= tfidf_vectorizer.transform(x_test.values.astype('U'))

#Passive Aggresive Classifier
print(" \nPassive Aggresive Classifier")
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(vec_train,y_train)
pac_y_pred=pac.predict(vec_test)

pac_score=accuracy_score(y_test,pac_y_pred)
pac_f1= f1_score(y_test,pac_y_pred,average="binary",pos_label="Real")
pac_recall = recall_score(y_test,pac_y_pred,average="binary",pos_label="Real")
pac_cm = confusion_matrix(y_test,pac_y_pred, labels=['Real','Fake'])
pac_pre= precision_score(y_test,pac_y_pred,average="binary",pos_label="Real")

print(f'PAC Accuracy: {round(pac_score*100,2)}%')
print("PAC F1 Score: ",pac_f1)
print("PAC Recall: ",pac_recall)
print("PAC Precision: ",pac_pre)
print("PAC CM:\n",pac_cm)

#PAC K-fold
pac_X=tfidf_vectorizer.transform(df['text'].values.astype('U'))
pac_k_scores = cross_val_score(pac, pac_X, df['label'].values, cv=5)
print(f'PAC K Fold Accuracy: {round(pac_k_scores.mean()*100,2)}%')

#Lojistic Regression
print(" \nLojistic Regression")
lr = LogisticRegression()
lr.fit(vec_train,y_train)
lr_y_pred = lr.predict(vec_test)

lr_score=accuracy_score(y_test,lr_y_pred)
lr_f1= f1_score(y_test,lr_y_pred,average="binary",pos_label="Real")
lr_recall = recall_score(y_test,lr_y_pred,average="binary",pos_label="Real")
lr_cm = confusion_matrix(y_test,lr_y_pred, labels=['Real','Fake'])
lr_pre= precision_score(y_test,lr_y_pred,average="binary",pos_label="Real")

print(f'LR Accuracy: {round(lr_score*100,2)}%')
print("LR F1 Score: ",lr_f1)
print("LR Recall: ",lr_recall)
print("LR Precision: ",lr_pre)

lr_cm = confusion_matrix(y_test,lr_y_pred, labels=['Real','Fake'])
print("LR CM \n",lr_cm)

#LR K-fold
lr_X=tfidf_vectorizer.transform(df['text'].values.astype('U'))
lr_k_score = cross_val_score(lr, lr_X, df['label'].values, cv=5)
print(f'LR K Fold Accuracy: {round(lr_k_score.mean()*100,2)}%')


#Decision Tree
print(" \nDecision Tree")
dt = DecisionTreeClassifier(random_state=0)
dt.fit(vec_train,y_train)
dt_y_pred = dt.predict(vec_test)

dt_score=accuracy_score(y_test,dt_y_pred)
dt_f1= f1_score(y_test,dt_y_pred,average="binary",pos_label="Real")
dt_recall = recall_score(y_test,dt_y_pred,average="binary",pos_label="Real")
dt_cm = confusion_matrix(y_test,dt_y_pred, labels=['Real','Fake'])
dt_pre= precision_score(y_test,dt_y_pred,average="binary",pos_label="Real")

print(f'DT Accuracy: {round(dt_score*100,2)}%')
print("DT F1 Score: ",dt_f1)
print("DT Recall: ",dt_recall)
print("DT Precision: ",dt_pre)
print("DT CM:\n",dt_cm)

#DT K-fold
dt_X=tfidf_vectorizer.transform(df['text'].values.astype('U'))
dt_k_score = cross_val_score(dt, dt_X, df['label'].values, cv=5)
print(f'DT K Fold Accuracy: {round(dt_k_score.mean()*100,2)}%')


#Multi. NB
print("\nNaive Bayes")
mnb = MultinomialNB()
mnb.fit(vec_train,y_train)
mnb_y_pred = mnb.predict(vec_test)

mnb_score=accuracy_score(y_test,mnb_y_pred)
mnb_cm = confusion_matrix(y_test,mnb_y_pred, labels=['Real','Fake'])
mnb_pre= precision_score(y_test,mnb_y_pred,average="binary",pos_label="Real")
mnb_f1= f1_score(y_test,mnb_y_pred,average="binary",pos_label="Real")
mnb_recall = recall_score(y_test,mnb_y_pred,average="binary",pos_label="Real")

print(f'MNB Accuracy: {round(mnb_score*100,2)}%')
print("MNB F1 Score: ",mnb_f1)
print("MNB Recall: ",mnb_recall)
print("MNB Precision: ",mnb_pre)
print("MNB CM:\n",mnb_cm)

#MNB K-fold
mnb_X=tfidf_vectorizer.transform(df['text'].values.astype('U'))
mnb_k_score = cross_val_score(mnb, mnb_X, df['label'].values, cv=5)
print(f'MNB K Fold Accuracy: {round(mnb_k_score.mean()*100,2)}%')


#Support Vector Machine
print(" \nSupport Vector Machine")
svcc = SVC(kernel="rbf")
svcc.fit(vec_train,y_train)
svcc_y_pred = svcc.predict(vec_test)

svcc_score=accuracy_score(y_test,svcc_y_pred)
svcc_cm = confusion_matrix(y_test,svcc_y_pred, labels=['Real','Fake'])
svcc_pre= precision_score(y_test,svcc_y_pred,average="binary",pos_label="Real")
svcc_f1= f1_score(y_test,svcc_y_pred,average="binary",pos_label="Real")
svcc_recall = recall_score(y_test,svcc_y_pred,average="binary",pos_label="Real")

print(f'SVM Accuracy: {round(svcc_score*100,2)}%')
print("SVM F1 Score: ",svcc_f1)
print("SVM Recall: ",svcc_recall)
print("SVM Precision: ",svcc_pre)
print("SVM CM \n",svcc_cm)

#SVM K-fold
svcc_X=tfidf_vectorizer.transform(df['text'].values.astype('U'))
svcc_k_score = cross_val_score(svcc, svcc_X, df['label'].values, cv=5)
print(f'SVM K Fold Accuracy: {round(svcc_k_score.mean()*100,2)}%')


