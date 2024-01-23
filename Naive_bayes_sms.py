# Import all the required libraries and modules
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# imbalanced-learn pipeline is being called in rather than a scikit-learn one.
# This is because we will be using SMOTE in our pipeline.
# pip install imblearn
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
# SMOTE - Synthetic Minority Over-sampling Technique
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.metrics as skmet
import joblib

# Loading the data set
data = pd.read_csv("D:/Data Scientist/Supervised Learning/Naive Bayes/Dataset/sms_raw_NB.csv", encoding = "ISO-8859-1")

# Mapping the type to numeric values 1 and 0. 
# This step is required for metric calculations in model evaluation phase.

data['spam'] = np.where(data.type == 'spam', 1, 0)

###############################################################################
# MySQL
# pip install mysql
import mysql.connector 
from sqlalchemy import create_engine
  

conn_string = ("mysql+mysqlconnector://{user}:{pw}@localhost/{db}"
               .format(user="root",  # MySQL username
                       pw="9130439933",      # MySQL password
                       db="sms_db"))  # MySQK database name

db = create_engine(conn_string)
conn = db.connect()

data.to_sql('sms_raw', con = conn, if_exists = 'replace', index = False)

conn.autocommit = True

###############################################################################
# Select query
sql = 'SELECT * from sms_raw'
email_data = pd.read_sql_query(sql, conn)


# Imbalance check
email_data.type.value_counts()
email_data.type.value_counts() / len(email_data.type)  # values in percentages

# alternatively
email_data.groupby(['type']).size()
email_data.groupby(['type']).size() / len(email_data.type)

# Data Split
email_train, email_test = train_test_split(email_data, test_size = 0.2, stratify = email_data[['spam']], random_state = 0) # StratifiedKFold is a variation of k-fold which returns stratified folds: each set contains approximately the same percentage of samples of each target class as the complete set.

# CountVectorizer
# Convert a collection of text documents to a matrix of token counts

countvectorizer = CountVectorizer(analyzer = 'word', stop_words = 'english')

###########################
# for illustrative purposes
s_sample = email_train.loc[email_train.text.str.len() < 50].sample(3, random_state = 35)
s_sample = s_sample.iloc[:, 0:2]

# Document Term Matrix with CountVectorizer (# returns 1D array)
s_vec = pd.DataFrame(countvectorizer.fit_transform(s_sample.values.ravel()).\
        toarray(), columns = countvectorizer.get_feature_names_out())

s_vec    
###########################    

# creating a matrix of token counts for the entire text document 
def split_into_words(i):
    return [word for word in i.split(" ")]

# Defining the preparation of email texts into word count matrix format - Bag of Words
emails_bow = CountVectorizer(analyzer = split_into_words).fit(email_data.text)

# Defining BOW for all messages
all_emails_matrix = emails_bow.transform(email_data.text)

# For training messages
train_emails_matrix = emails_bow.transform(email_train.text)

# For testing messages
test_emails_matrix = emails_bow.transform(email_test.text)



# We will use SMOTE technique to handle class imbalance.
# Oversampling can be a good option when we have class imbalance.
# Due to this our model will perform poorly in capturing variation in a class
# because we have too few instances of that class, relative to one or more other classes.

# SMOTE: Is an approach is to oversample (duplicating examples) the minority class
# This is a type of data augmentation for the minority class and is referred 
# to as the Synthetic Minority Oversampling Technique, or SMOTE for short.
smote = SMOTE(random_state = 0)

# Transform the dataset
X_train, y_train = smote.fit_resample(train_emails_matrix, email_train.spam)

y_train.unique()
y_train.values.sum()   # Number of '1's
y_train.size - y_train.values.sum()  # Number of '0's
# The data is now balanced

# Multinomial Naive Bayes
classifier_mb = MultinomialNB()   #vanilla model with default parameters
classifier_mb.fit(X_train, y_train)

# Evaluation on Test Data
test_pred_m = classifier_mb.predict(test_emails_matrix)

pd.crosstab(email_test.spam, test_pred_m)

# Accuracy
accuracy_test_m = np.mean(test_pred_m == email_test.spam)
accuracy_test_m

# or alternatively
skmet.accuracy_score(email_test.spam, test_pred_m) 


# Training Data accuracy
train_pred_m = classifier_mb.predict(train_emails_matrix)

pd.crosstab(email_train.spam, train_pred_m)

# Accuracy
accuracy_train_m = np.mean(train_pred_m == email_train.spam)
accuracy_train_m

skmet.accuracy_score(email_train.spam, train_pred_m) 


############################################
# Model Tuning - Hyperparameter optimization

# Multinomial Naive Bayes changing default alpha for laplace smoothing
# if alpha = 0 then no smoothing is applied and the default alpha parameter is 1
# the smoothing process mainly solves the emergence of zero probability problem in the dataset.

# formula: 
# P(w|spam) = (num of spam with w + alpha)/(Total num of spam emails + K(alpha))
# K = total num of words in the email to be classified

mnb_lap = MultinomialNB(alpha = 5)

mnb_lap.fit(X_train, y_train)

# Evaluation on Test Data after applying laplace
test_pred_lap = mnb_lap.predict(test_emails_matrix)

pd.crosstab(test_pred_lap, email_test.type)

accuracy_test_lap = np.mean(test_pred_lap == email_test.spam)
accuracy_test_lap

skmet.accuracy_score(email_test.spam, test_pred_lap) 


# Training Data accuracy
train_pred_lap = mnb_lap.predict(train_emails_matrix)

pd.crosstab(train_pred_lap, email_train.spam)

accuracy_train_lap = np.mean(train_pred_lap == email_train.spam)
accuracy_train_lap

skmet.accuracy_score(email_train.spam, train_pred_lap) 


# Metrics
print("accuracy: %.2f, sensitivity: %.2f, specificity: %.2f, precision: %.2f"  %
  (skmet.accuracy_score(email_test.spam.ravel(), test_pred_lap),
  skmet.recall_score(email_test.spam.ravel(), test_pred_lap),
  skmet.recall_score(email_test.spam.ravel(), test_pred_lap, pos_label = 0),
  skmet.precision_score(email_test.spam.ravel(), test_pred_lap)))


# Confusion Matrix - Heat Map
cm = skmet.confusion_matrix(email_test.spam, test_pred_lap)
cmplot = skmet.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['Not Spam', 'Spam'])
cmplot.plot()
cmplot.ax_.set(title = 'Spam Detection Confusion Matrix', 
               xlabel = 'Predicted Value', ylabel = 'Actual Value')




# Saving the Best Model using Pipelines

# Building the Pipeline
# Preparing a naive bayes model on training data set 
nb = MultinomialNB(alpha = 5)


# Defining Pipeline
pipe1 = make_pipeline(countvectorizer, smote, nb)

# Fit the train data
processed = pipe1.fit(email_train.text.ravel(), email_train.spam.ravel())


# Save the trained model
joblib.dump(processed, 'processed1')


# load the saved model for predictions
model = joblib.load('processed1')

# Predictions
test_pred = model.predict(email_test.text.ravel())

# Evaluation on Test Data with Metrics
# Confusion Matrix
pd.crosstab(email_test.spam, test_pred)

# Accuracy
skmet.accuracy_score(email_test.spam, test_pred)

# Metrics
print("accuracy: %.2f, sensitivity: %.2f, specificity: %.2f, precision: %.2f"  %
  (skmet.accuracy_score(email_test.spam.ravel(), test_pred),
  skmet.recall_score(email_test.spam.ravel(), test_pred),
  skmet.recall_score(email_test.spam.ravel(), test_pred, pos_label = 0),
  skmet.precision_score(email_test.spam.ravel(), test_pred)))

# Confusion Matrix - Heat Map
cm = skmet.confusion_matrix(email_test.spam, test_pred)
cmplot = skmet.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['Not Spam', 'Spam'])
cmplot.plot()
cmplot.ax_.set(title = 'Spam Detection Confusion Matrix', 
               xlabel = 'Predicted Value', ylabel = 'Actual Value')



