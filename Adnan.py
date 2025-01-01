#Basic libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Load training/validation data
trainval_data = pd.read_csv('train.csv')
#Reading dataset
trainval_data.head()
#Check for any empty columns
trainval_data.apply(lambda x:sum(x.isnull()), axis=0)
#The data is required in numbers, therefore replace the yes/no in floods coloumn by 1/0
trainval_data['Flood'].replace(['Yes','No'],[1,0],inplace=True)
trainval_data.head()
from sklearn.preprocessing import LabelEncoder
# taking 'SUBDIVISION' column name in your dataset to encode
label_encoder = LabelEncoder()
trainval_data['subdivision_encoded'] = label_encoder.fit_transform(trainval_data['SUBDIVISION'])
#look at the dataset
trainval_data.head(10) 
#moving the encoded subdivision column infront to easily separate target variable in trainval_data
column_order = ['subdivision_encoded', 'SUBDIVISION', 'YEAR', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL','AUG', 'SEP', 'OCT', 'NOV', 'DEC', 'ANNUAL', 'Flood']
# Rearrange columns based on 'column_order'
trainval_data = trainval_data[column_order]
trainval_data.head()
trainval_data.drop('SUBDIVISION', axis=1, inplace=True)
#we only require numerical data, so removing string datatype (already encoded)
#Seperate the data for prediction, considering only numeric data
X = trainval_data.iloc[:,0:15]
X.head()
X.tail()
# Load testing data
test_data = pd.read_csv('maintest.csv')
test_data.head()
#Check for any colomns is left empty
test_data.apply(lambda x:sum(x.isnull()), axis=0)
#The data is required in numbers, therefore replace the yes/no in floods coloumn by 1/0
test_data['FLOOD'].replace(['YES','NO'],[1,0],inplace=True)
test_data.head()
#encoding string to integer
label_encoder = LabelEncoder()
test_data['subdivision_encoded'] = label_encoder.fit_transform(test_data['SUBDIVISIONS'])
test_data.head(10)
#moving the encoded subdivision column infront to easily separate target variable in test_data
column_order = ['subdivision_encoded', 'SUBDIVISIONS', 'YEAR', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL','AUG', 'SEP', 'OCT', 'NOV', 'DEC', 'ANNUAL', 'FLOOD']
# Rearrange columns based on 'column_order'
test_data = test_data[column_order]
test_data.head()
test_data.drop('SUBDIVISIONS', axis=1, inplace=True) #removing string datatype as already encoded as integer
test_data.head(10)
#Explore how rainfall index varies during rainy season for trainval_data
%matplotlib inline
c = trainval_data[['JUN','JUL','AUG','SEP']]
c.hist()
plt.show()
#Explore how rainfall index varies during rainy season for test dataset
%matplotlib inline
c = test_data[['JUN','JUL','AUG','SEP']]
c.hist()
plt.show()
#Data might be widely distributed so let's scale it between 0 and 1
from sklearn import preprocessing
minmax = preprocessing.MinMaxScaler(feature_range=(0,1))
minmax.fit(X).transform(X)
from sklearn.model_selection import train_test_split

# Split the training+validation data into training (80%) and validation (20%)
X_trainval, X_val, y_trainval, y_val = train_test_split(
    trainval_data.drop('Flood', axis=1),  # Features
    trainval_data['Flood'],  # Target variable
    test_size=0.2,  # Validation size
    random_state=42
)

# Remaining training data after splitting
X_train, y_train = X_trainval, y_trainval
#Train and validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
# Initialize KNN model
knn = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors

# Train the model on the training data
knn.fit(X_train, y_train)

# Predict on the validation set
y_pred_val = knn.predict(X_val)
# Evaluate performance on validation set
accuracy_val = accuracy_score(y_val, y_pred_val)
print("Validation Accuracy:", accuracy_val)
# Additional evaluation metrics or reports for validation set
report_val = classification_report(y_val, y_pred_val)
print("Validation Classification Report:\n", report_val)
# Test the trained model on the separate test set
# Separate test data
X_test, y_test = test_data.drop('FLOOD', axis=1), test_data['FLOOD']

# Predict on the test set
y_pred_test = knn.predict(X_test)
# Evaluate performance on the test set
accuracy_test = accuracy_score(y_test, y_pred_test)
print("Test Accuracy:", accuracy_test)
# Additional evaluation metrics or reports for test set
report_test = classification_report(y_test, y_pred_test)
print("Test Classification Report:\n", report_test)
from sklearn.linear_model import LogisticRegression
# Initialize Logistic Regression classifier
log_reg = LogisticRegression(max_iter=1000)

# Train the model on the training data
log_reg.fit(X_train, y_train)

# Predict on the validation set
y_pred_val_log_reg = log_reg.predict(X_val)
# Evaluate performance on validation set
accuracy_val_log_reg = accuracy_score(y_val, y_pred_val_log_reg)
print("Validation Accuracy:", accuracy_val_log_reg)
# Additional evaluation metrics or reports for validation set
report_val_log_reg = classification_report(y_val, y_pred_val_log_reg)
print("Validation Classification Report:\n", report_val_log_reg)
# Test the trained model on the separate test set
y_pred_test_log_reg = log_reg.predict(X_test)
# Evaluate performance on the test set
accuracy_test_log_reg = accuracy_score(y_test, y_pred_test_log_reg)
print("Test Accuracy:", accuracy_test)
# Additional evaluation metrics or reports for test set
report_test_log_reg = classification_report(y_test, y_pred_test_log_reg)
print("Test Classification Report:\n", report_test_log_reg)
from sklearn.tree import DecisionTreeClassifier
# Initialize Decision Tree Classifier
decision_tree = DecisionTreeClassifier(random_state=42)

# Train the model on the training set
decision_tree.fit(X_train, y_train)
# Validate on the validation set
y_pred_val_dec_tre = decision_tree.predict(X_val)
# Evaluate performance on validation set
accuracy__dec_tre = accuracy_score(y_val, y_pred_val_dec_tre)
print("Validation Accuracy:", accuracy__dec_tre)
# Additional evaluation metrics or reports for validation set
report__dec_tre = classification_report(y_val, y_pred_val_dec_tre)
print("Validation Classification Report:\n", report__dec_tre)
# Test the trained model on the separate test set
y_pred_test_dec_tre = decision_tree.predict(X_test)
# Evaluate performance on the test set
accuracy_dec_tre = accuracy_score(y_test, y_pred_test_dec_tre)
print("Test Accuracy:", accuracy_dec_tre)
# Additional evaluation metrics or reports for test set
report_dec_tre = classification_report(y_test, y_pred_test_dec_tre)
print("Test Classification Report:\n", report_dec_tre)
from sklearn.ensemble import RandomForestClassifier
# Initialize Random Forest Classifier
random_forest = RandomForestClassifier(random_state=42) # You can set other hyperparameters as ne
# Train the model on the training set
random_forest.fit(X_train, y_train)
# Validate on the validation set
y_pred_val_ran_for = random_forest.predict(X_val)
# Evaluate performance on validation set
accuracy_ran_for = accuracy_score(y_val, y_pred_val_ran_for)
print("Validation Accuracy:", accuracy_ran_for)
# Additional evaluation metrics or reports for validation set
report_ran_for = classification_report(y_val, y_pred_val_ran_for)
print("Validation Classification Report:\n", report_ran_for)
# Test the trained model on the separate test set
y_pred_test_ran_for = random_forest.predict(X_test)
# Evaluate performance on the test set
accuracy_ran_for = accuracy_score(y_test, y_pred_test_ran_for)
print("Test Accuracy:", accuracy_ran_for)
# Additional evaluation metrics or reports for test set
report_ran_for = classification_report(y_test, y_pred_test_ran_for)
print("Test Classification Report:\n", report_ran_for)
from sklearn.svm import SVC
svm_model = SVC(kernel='linear', C=1.0)  # Example with a linear kernel and regularization parameter C=1.0
svm_model.fit(X_train, y_train)

# Predict on the validation set
y_pred_val_svm = svm_model.predict(X_val)
# Evaluate performance on validation set
accuracy_val_svm = accuracy_score(y_val, y_pred_val_svm)
print("Validation Accuracy:", accuracy_val_svm)
# Additional evaluation metrics or reports for validation set
report_val_svm = classification_report(y_val, y_pred_val_svm)
print("Validation Classification Report:\n", report_val_svm)
# Test the trained model on the separate test set
y_pred_test_svm = svm_model.predict(X_test)
# Evaluate performance on the test set
accuracy_test_svm = accuracy_score(y_test, y_pred_test_svm)
print("Test Accuracy:", accuracy_test_svm)
# Additional evaluation metrics or reports for test set
report_test_svm = classification_report(y_test, y_pred_test_svm)
print("Test Classification Report:\n", report_test_svm)
from sklearn.naive_bayes import GaussianNB
naive_bayes_model = GaussianNB()
naive_bayes_model.fit(X_train, y_train)

# Predict on the validation set
y_pred_val_nb = naive_bayes_model.predict(X_val)
# Evaluate performance on validation set
accuracy_val_nb = accuracy_score(y_val, y_pred_val_nb)
print("Validation Accuracy:", accuracy_val_nb)
# Additional evaluation metrics or reports for validation set
report_val_nb = classification_report(y_val, y_pred_val_nb)
print("Validation Classification Report:\n", report_val_nb)
# Test the trained model on the separate test set
y_pred_test_nb = naive_bayes_model.predict(X_test)
# Evaluate performance on the test set
accuracy_test_nb = accuracy_score(y_test, y_pred_test_nb)
print("Test Accuracy:", accuracy_test_nb)
# Additional evaluation metrics or reports for test set
report_test_nb = classification_report(y_test, y_pred_test_nb)
print("Test Classification Report:\n", report_test_nb)
models = [knn, log_reg, decision_tree, random_forest,svm_model ,naive_bayes_model ]
model_names = ['KNN', 'Logistic Regression', 'Decision Tree', 'Random Forest','SVM','Naive Bayes']
for model, name in zip(models, model_names):
 # Train the model on the training set
 model.fit(X_train, y_train)
 
 # Validate the model on the validation set
 y_pred_val = model.predict(X_val)
 
 # Evaluate model performance on validation set
 accuracy_val = accuracy_score(y_val, y_pred_val)
 # Print or store the validation results for comparison
 print(f"{name} Validation Accuracy: {accuracy_val}")
  # Validation and test accuracy scores for different models
models = ['KNN', 'Logistic Regression', 'Decision Tree', 'Random Forest','SVM','Naive Bayes']
val_scores = [0.990, 0.992, 1.0, 1.0, 1.0, 0.893]
test_scores = [0.991, 0.991, 1.0, 1.0, 1.0, 0.947]

# Create a DataFrame to compare accuracy scores
accuracy_df = pd.DataFrame({'Model': models, 'Validation Accuracy': val_scores, 'Test Accuracy': test_scores})

# Display the comparison table
print(accuracy_df)
#plot line graph
models = ['KNN', 'Logistic Regression', 'Decision Tree', 'Random Forest','SVM','Naive Bayes']
val_scores = [0.993, 0.986, 1.0, 1.0, 1.0, 0.914]
test_scores = [0.991, 0.991, 1.0, 1.0, 1.0, 0.947]

plt.plot(models, val_scores, marker='o', label='Validation Accuracy')
plt.plot(models, test_scores, marker='o', label='Test Accuracy')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Validation and Test Accuracy Scores for the Models')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
new_data= pd.read_csv('2022rainfall.csv')
#Explore how rainfall index varies during rainy season for this unknown data
%matplotlib inline
c = new_data[['JUN','JUL','AUG','SEP']]
c.hist()
plt.show()
new_data.head()
#encoding string to integer
subdiv_encoder = LabelEncoder()
new_data['subdivision_encoded'] = subdiv_encoder.fit_transform(new_data['SUBDIVISIONS'])
#moving the encoded subdivision column infront to easily separate target variable in test_data
column_order = ['subdivision_encoded', 'SUBDIVISIONS', 'YEAR', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL','AUG', 'SEP', 'OCT', 'NOV', 'DEC', 'ANNUAL', 'FLOOD']
# Rearrange columns based on 'column_order'
new_data = new_data[column_order]
new_data.head()
#The data is required in numbers, therefore replace the yes/no in floods coloumn by 1/0
new_data['FLOOD'].replace(['YES','NO'],[1,0],inplace=True)
new_data.head()
testing = new_data.drop('FLOOD', axis=1)
checkvar = new_data['FLOOD']
testing = testing.drop('SUBDIVISIONS', axis=1)
testing.head()
#predicting chances of flood in 0/1
validity_dec_tree = decision_tree.predict(testing)
print(validity_dec_tree)
# Evaluate performance
accuracy_testing = accuracy_score(checkvar, validity_dec_tree)
print("Validation Accuracy:", accuracy_testing)
testing_svm = svm_model.predict(testing)
print(testing_svm)
# Evaluate performance
accuracy_svm = accuracy_score(checkvar, testing_svm)
print("Validation Accuracy:", accuracy_svm)
import matplotlib.pyplot as plt

models = ['Decision Tree', 'SVM']
accuracy_scores = [1.0, 0.997]

plt.figure(figsize=(8, 6))
plt.plot(models, accuracy_scores, marker='o', linestyle='-', color='orange')

plt.title('Accuracy Scores of Different Models')
plt.xlabel('Models')
plt.ylabel('Accuracy Score')
plt.ylim(0.95, 1.005)  # Adjust the y-axis limits for better visualization
plt.grid(True)

# Adding data labels
for i, score in enumerate(accuracy_scores):
    plt.text(i, score, f'{score:.3f}', ha='center', va='bottom')

plt.show()
