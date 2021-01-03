import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import f1_score , accuracy_score, precision_recall_curve, plot_precision_recall_curve, recall_score
import matplotlib.pyplot as plt

def prep_data(file):
  df = pd.read_csv(file)
  df['category_id'] = df.category.factorize()[0]

  category_id_df = df[['category', 'category_id']].drop_duplicates().sort_values('category_id')
  category_to_id = dict(category_id_df.values)
  id_to_category = dict(category_id_df[['category_id', 'category']].values)
  
  tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
  
  features = tfidf.fit_transform(df.article).toarray()
  labels = df.category_id

  return features, labels, df
  

def log_reg(f, l, t_size):
  model = LogisticRegression(random_state=0)
  X_train, X_test, y_train, y_test = train_test_split(f, l, test_size=t_size, random_state=0)
  model.fit(X_train, y_train)
  y_pred_proba = model.predict_proba(X_test)
  y_pred = model.predict(X_test)
  
  conf_mat = confusion_matrix(y_test, y_pred)
  f1 =  f1_score(y_test, y_pred, average='macro')
  
  precision = accuracy_score(y_test, y_pred)
  recall = recall_score (y_test, y_pred, average ='macro')
  
  return conf_mat, f1, precision,recall
  
def knn(f, l, t_size, neighbours = 5):
  model = KNeighborsClassifier(n_neighbors=neighbours)
  X_train, X_test, y_train, y_test = train_test_split(f, l, test_size=t_size, random_state=0)
  model.fit(X_train, y_train)
  y_pred_proba = model.predict_proba(X_test)
  y_pred = model.predict(X_test)
  
  conf_mat = confusion_matrix(y_test, y_pred)
  f1 =  f1_score(y_test, y_pred, average='macro')
  
  precision = accuracy_score(y_test, y_pred)
  recall = recall_score (y_test, y_pred, average ='macro')
  
  return conf_mat, f1, precision,recall




def svm_clf(f, l, t_size):
  model = SVC()
  X_train, X_test, y_train, y_test = train_test_split(f, l, test_size=t_size, random_state=0)
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  
  conf_mat = confusion_matrix(y_test, y_pred)
  f1 =  f1_score(y_test, y_pred, average='macro')
  
  precision = accuracy_score(y_test, y_pred)
  recall = recall_score (y_test, y_pred, average ='macro')
  
  return conf_mat, f1, precision,recall


def gnb_clf(f, l, t_size):
  model = GaussianNB()
  X_train, X_test, y_train, y_test = train_test_split(f, l, test_size=t_size, random_state=0)
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  
  conf_mat = confusion_matrix(y_test, y_pred)
  f1 =  f1_score(y_test, y_pred, average='macro')
  
  precision = accuracy_score(y_test, y_pred)
  recall = recall_score (y_test, y_pred, average ='macro')
  
  return conf_mat, f1, precision,recall



def dtrees_clf(f, l, t_size, depth = 5):
  model = DecisionTreeClassifier(max_depth= depth)
  X_train, X_test, y_train, y_test = train_test_split(f, l, test_size=t_size, random_state=0)
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  
  conf_mat = confusion_matrix(y_test, y_pred)
  f1 =  f1_score(y_test, y_pred, average='macro')
  
  precision = accuracy_score(y_test, y_pred)
  recall = recall_score (y_test, y_pred, average ='macro')
  
  return conf_mat, f1, precision,recall


def rf_clf(f, l, t_size, depth =5 , estimators =10, features=1):
  model = RandomForestClassifier(max_depth=depth, n_estimators= estimators, max_features= features)
  X_train, X_test, y_train, y_test = train_test_split(f, l, test_size=t_size, random_state=0)
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  
  conf_mat = confusion_matrix(y_test, y_pred)
  f1 =  f1_score(y_test, y_pred, average='macro')
  
  precision = accuracy_score(y_test, y_pred)
  recall = recall_score (y_test, y_pred, average ='macro')
  
  return conf_mat, f1, precision,recall












