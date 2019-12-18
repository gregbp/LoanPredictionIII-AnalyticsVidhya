import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, average_precision_score, f1_score, recall_score
import numpy as np


import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)


# Evaluating our model by splitting the train set into train and test sets (70-30)
def ModelEvaluation(train):

    # New train and test sets
    X = train.drop('Loan_Status', axis=1)
    y = train['Loan_Status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

    # Dictionary that stores the algorithm name (key) and its precision (value)
    accuracy_list = {}

    # Dataframe that stores the metric results for every algorithm
    data = {'F1_score': [0.0, 0.0, 0.0, 0.0, 0.0],
            'Recall Score': [0.0, 0.0, 0.0, 0.0, 0.0],
            'Average Precision Score': [0.0, 0.0, 0.0, 0.0, 0.0],
            'Accuracy Score': [0.0, 0.0, 0.0, 0.0, 0.0],
            }
    metrics = pd.DataFrame(data)
    metrics.index = ['Decision Tree', 'Random Forest', 'KNN', 'SVM', 'Logistic Regression']

    # DecisionTree
    dtree = DecisionTreeClassifier()
    dtree.fit(X_train, y_train)
    predictions = dtree.predict(X_test)

    accuracy_list.update({'Decision Tree': accuracy_score(y_test, predictions)})

    metrics.iloc[0] = [f1_score(y_test, predictions), recall_score(y_test, predictions),
                       average_precision_score(y_test, predictions), accuracy_score(y_test, predictions)]

    print('\n\nDecisionTree')
    print(confusion_matrix(y_test,predictions), '\n\n')

    # RandomForest
    rfc = RandomForestClassifier(n_estimators=100)
    rfc.fit(X_train, y_train)
    predictions = rfc.predict(X_test)

    accuracy_list.update({'Random Forest': accuracy_score(y_test, predictions)})

    metrics.iloc[1] = [f1_score(y_test, predictions), recall_score(y_test, predictions),
                       average_precision_score(y_test, predictions), accuracy_score(y_test, predictions)]

    print('RandomForest')
    print(confusion_matrix(y_test, predictions), '\n\n')

    # KNN
    # Finding the best K
    max_acc_scr = -1.0
    max_k = -1
    for k in range(1, 30):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)

        acc_scr = accuracy_score(y_test, predictions)

        if acc_scr > max_acc_scr:
            max_acc_scr = acc_scr
            max_k = k

    # Fitting a model using the found best K
    knn = KNeighborsClassifier(n_neighbors=max_k)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)

    accuracy_list.update({'KNN': accuracy_score(y_test, predictions)})

    metrics.iloc[2] = [f1_score(y_test, predictions), recall_score(y_test, predictions),
                       average_precision_score(y_test, predictions), accuracy_score(y_test, predictions)]

    print('KNN')
    print(confusion_matrix(y_test, predictions), '\n\n')

    # SVM
    model = SVC()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy_list.update({'SVM': accuracy_score(y_test, predictions)})

    metrics.iloc[3] = [f1_score(y_test, predictions), recall_score(y_test, predictions),
                       average_precision_score(y_test, predictions), accuracy_score(y_test, predictions)]

    print('SVM')
    print(confusion_matrix(y_test, predictions), '\n\n')

    # LogisticRegression
    logmodel = LogisticRegression(solver='lbfgs')
    logmodel.fit(X_train, y_train)
    predictions = logmodel.predict(X_test)

    accuracy_list.update({'Logistic Regression': accuracy_score(y_test, predictions)})

    metrics.iloc[4] = [f1_score(y_test, predictions), recall_score(y_test, predictions),
                       average_precision_score(y_test, predictions), accuracy_score(y_test, predictions)]

    print('LogisticRegression')
    print(confusion_matrix(y_test, predictions), '\n\n')



    # Printing the metric results for every algorithm
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print("\n\n\tmetrics DF inside function\n", metrics, "\n\n")
    print('accuracy_list -> ', accuracy_list)

    # Finding the name of the best algorithm (using finally accuracy as a measure)
    best_alg = max(accuracy_list, key=accuracy_list.get)
    print('best_alg ', best_alg)

    # Returning the name of best algorithm (best_alg), its f1 score
    # (accuracy_list[best_alg]) and best K for KNN algorithm (max_k)
    return best_alg, accuracy_list[best_alg], max_k



# Finding the mean number of dependent members for each marital status (married and not married)
def find_mean_married_dependents(sample, marital_status):

    depDF = (sample[sample['Married'] == marital_status]['Dependents']).to_frame()
    depDF.dropna(inplace=True)

    zeros = depDF[depDF['Dependents'] == '0'].count()
    ones = depDF[depDF['Dependents'] == '1'].count()
    twos = depDF[depDF['Dependents'] == '2'].count()
    threes = depDF[depDF['Dependents'] == '3+'].count()

    total = zeros + ones + twos + threes
    mean_dep = float((ones + twos * 2 + threes * 3) / total)

    return round(mean_dep)



# Filling NAs in 'Dependents' by using the mean calculated on 'find_mean_married_dependents(sample, marital_status)'
def fillNA_Dependents_based_on_Married(x):

    if pd.isnull(x['Dependents']):
        if x['Married'] == 'No':
            x['Dependents'] = mean_dep_NotMarried
        elif x['Married'] == 'Yes':
            x['Dependents'] = mean_dep_Married

    return x['Dependents']



# Filling NAs in 'LoanAmount' column based on 'Education' and 'Self_Employed' columns
def fillNA_LoanAmount(x):
    if pd.isnull(x['LoanAmount']):
        if x['Education'] == 'Graduate':
            if x['Self_Employed'] == 'No':
                return pivot_table.loc[0, 0]
            else:
                return pivot_table.loc[1, 0]
        else:
            if x['Self_Employed'] == 'No':
                return pivot_table.loc[2, 0]
            else:
                return pivot_table.loc[3, 0]
    else:
        return x['LoanAmount']



# Using KKN imputation in order to fill NA value of 'Gender', 'Self_Employed', 'Loan_Amount_Term'
# (both in train and test sets) and 'Credit_History' (only in test set)
def KNN_Imputation(init_sample, attr):

    # init_sample -> original dataframe
    # attr -> the name of the column that will be filled by the function

    # Copying original dataframe in order to avoid unwanted changes
    sample = init_sample.copy()

    # Dropping NA values in 'sample' dataframe
    droppedNAcols = list(sample.columns)
    droppedNAcols.remove(attr)
    sample.dropna(subset=droppedNAcols, how='any', inplace=True)
    NArows = pd.isna(sample[attr])
    filled_rows = pd.isna(sample[attr]) == False

    # Creating the necessary dataframes for the KNN
    train = sample[filled_rows].dropna()
    X_train = train[filled_rows].drop(attr, axis=1)
    y_train = train[filled_rows][attr]
    X_test = sample[NArows].drop(attr, axis=1)

    # KNN
    for k in range(10, 20):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)

    # Indeces of the completed NA rows
    idxs = list(X_test.index)

    # Indeces and predicted values for each NA row
    preds = dict(zip(idxs, predictions))

    return preds



# ------ MAIN PROGRAM

# Directory for the input & output files
dir_path = os.path.dirname(os.path.realpath(__file__))
inout_path = dir_path+'\\inoutput'

# Initial train and test dataframes
train = pd.read_csv(inout_path + "\\train_ctrUa4K.csv")
test = pd.read_csv(inout_path + "\\test_lAUu6dG.csv")

# Keeping in the train set the same columns that exist in the test set + 'Loan_Status' column
test_cols = test.columns.tolist()
test_cols.append('Loan_Status')
train = train[test_cols]

# Dropping 'Loan_ID' column as useless
train.drop('Loan_ID', axis=1, inplace=True)


# ----- TRAIN SET

# --- NA treatment part 1 - 'Credit_History', 'Dependents', 'LoanAmount'

# 'Credit_History' - filled with Loan_Status
train['Credit_History'].fillna(train['Loan_Status'], inplace=True)

# 'Married' - the NA rows are dropped
train.dropna(subset=['Married'], inplace=True, axis=0)

# Dependents - filled based on Married
mean_dep_Married = find_mean_married_dependents(train, 'Yes')
mean_dep_NotMarried = find_mean_married_dependents(train, 'No')
train['Dependents'] = train.apply(lambda x: fillNA_Dependents_based_on_Married(x), axis=1)

# LoanAmount - filled based on Education and Self_Employed
pivot_table = train.pivot_table(index=['Education', 'Self_Employed'], values='LoanAmount',
                                aggfunc=np.mean).sum(axis=1).reset_index()
train['LoanAmount'] = train.apply(lambda x: fillNA_LoanAmount(x), axis=1)


# --- Converting object columns to numerical - 'Married', 'Education', 'Self_Employed', 'Credit_History', 'Loan_Status',
# 'Dependents', 'Property_Area'

# Converting "binary" classes to 0-1
train['Married'] = train['Married'].apply(lambda row: 1 if row == 'Yes' else 0)

train['Education'] = train['Education'].apply(lambda row: 1 if row == 'Graduate' else 0)

train['Self_Employed'] = train['Self_Employed'].apply(lambda row: 1 if row == 'Yes' else 0 if row == 'No' else np.nan)

train['Credit_History'] = train['Credit_History'].apply(lambda row: 1 if row == 'Y' else 0)

train['Loan_Status'] = train['Loan_Status'].apply(lambda row: 1 if row == 'Y' else 0)

# Creating a new column for each value of'Dependents' and 'Property_Area'
train['0 Dependents'] = train['Dependents'].apply(lambda row: 1 if row == '0' else 0)
train['1 Dependents'] = train['Dependents'].apply(lambda row: 1 if row == '1' else 0)
train['2 Dependents'] = train['Dependents'].apply(lambda row: 1 if row == '2' else 0)
train['3+ Dependents'] = train['Dependents'].apply(lambda row: 1 if row == '3+' else 0)

train['Urban'] = train['Property_Area'].apply(lambda row: 1 if row == 'Urban' else 0)
train['Rural'] = train['Property_Area'].apply(lambda row: 1 if row == 'Rural' else 0)
train['Semiurban'] = train['Property_Area'].apply(lambda row: 1 if row == 'Semiurban' else 0)

# Getting rid of the "original" columns 'Dependents', 'Property_Area'
dropped_cols = ['Dependents', 'Property_Area']
train.drop(dropped_cols, axis=1, inplace=True)

# Adding 'ApplicantIncome', 'CoapplicantIncome' to the 'Total_Income' column
train['Total_Income'] = train['ApplicantIncome'] + train['CoapplicantIncome']
dropped_cols = ['ApplicantIncome', 'CoapplicantIncome']
train.drop(dropped_cols, axis=1, inplace=True)


# --- NA treatment part 2 - 'Loan_Amount_Term', 'Self_Employed', 'Gender' (filled using KNN)

prds = KNN_Imputation(train, 'Gender')
train['Gender'].fillna(prds, inplace=True)

# Creating two new columns for each value of 'Gender' and dropping 'Gender'
train['Male'] = train['Gender'].apply(lambda row: 1 if row == 'Male' else 0)
train['Female'] = train['Gender'].apply(lambda row: 1 if row == 'Female' else 0)
train.drop('Gender', inplace=True, axis=1)

prds = KNN_Imputation(train, 'Self_Employed')
train['Self_Employed'].fillna(prds, inplace=True)

prds = KNN_Imputation(train, 'Loan_Amount_Term')
train['Loan_Amount_Term'].fillna(prds, inplace=True)

# Dropping rows that have more than one NA column and can't be filled using KNN
train.dropna(inplace=True)



# ----- TEST SET

# Dropping 'Loan_ID' column and keeping it for future use
loanID_df = pd.DataFrame(test['Loan_ID'].copy())
test.drop('Loan_ID', axis=1, inplace=True)

# --- NA treatment part 1 - 'Dependents', 'LoanAmount'

# Dependents - filled based on Married
mean_dep_Married = find_mean_married_dependents(test, 'Yes')
mean_dep_NotMarried = find_mean_married_dependents(test, 'No')
test['Dependents'] = test.apply(lambda x: fillNA_Dependents_based_on_Married(x), axis=1)

# 'LoanAmount' - filled based on Education and Self_Employed
pivot_table = test.pivot_table(index=['Education', 'Self_Employed'], values='LoanAmount', aggfunc=np.mean).sum(axis=1).reset_index()
test['LoanAmount'] = test.apply(lambda x: fillNA_LoanAmount(x), axis=1)


# --- Converting object columns to numerical - 'Married', 'Education', 'Self_Employed', 'Dependents', 'Property_Area'

# Converting "binary" classes to 0-1
test['Married'] = test['Married'].apply(lambda row: 1 if row == 'Yes' else 0)

test['Education'] = test['Education'].apply(lambda row: 1 if row == 'Graduate' else 0)

test['Self_Employed'] = test['Self_Employed'].apply(lambda row: 1 if row == 'Yes' else 0 if row == 'No' else np.nan)

# Creating a new column for each value of'Dependents' and 'Property_Area'
test['0 Dependents'] = test['Dependents'].apply(lambda row: 1 if row == '0' else 0)
test['1 Dependents'] = test['Dependents'].apply(lambda row: 1 if row == '1' else 0)
test['2 Dependents'] = test['Dependents'].apply(lambda row: 1 if row == '2' else 0)
test['3+ Dependents'] = test['Dependents'].apply(lambda row: 1 if row == '3+' else 0)

test['Urban'] = test['Property_Area'].apply(lambda row: 1 if row == 'Urban' else 0)
test['Rural'] = test['Property_Area'].apply(lambda row: 1 if row == 'Rural' else 0)
test['Semiurban'] = test['Property_Area'].apply(lambda row: 1 if row == 'Semiurban' else 0)

# Getting rid of the "original" columns 'Dependents', 'Property_Area'
dropped_cols = ['Dependents', 'Property_Area']
test.drop(dropped_cols, axis=1, inplace=True)

# Adding 'ApplicantIncome', 'CoapplicantIncome' to the 'Total_Income' column
test['Total_Income'] = test['ApplicantIncome'] + test['CoapplicantIncome']
dropped_cols = ['ApplicantIncome', 'CoapplicantIncome']
test.drop(dropped_cols, axis=1, inplace=True)


# --- NA treatment part 2 - 'Gender', 'Self_Employed', 'Loan_Amount_Term', 'Credit_History' (filled using KNN)

prds = KNN_Imputation(test, 'Gender')
test['Gender'].fillna(prds, inplace=True)
test['Male'] = test['Gender'].apply(lambda row: 1 if row == 'Male' else 0)
test['Female'] = test['Gender'].apply(lambda row: 1 if row == 'Female' else 0)
test.drop('Gender', inplace=True, axis=1)

prds = KNN_Imputation(test, 'Self_Employed')
test['Self_Employed'].fillna(prds, inplace=True)

prds = KNN_Imputation(test, 'Loan_Amount_Term')
test['Loan_Amount_Term'].fillna(prds, inplace=True)

prds = KNN_Imputation(test, 'Credit_History')
test['Credit_History'].fillna(prds, inplace=True)

# Filling rows that have more than one NA column with the most frequent value of each feature - These rows can't be filled using KNN
test['Self_Employed'].fillna(value=float(test['Self_Employed'].mode()), inplace=True)
test['Loan_Amount_Term'].fillna(value=float(test['Loan_Amount_Term'].mode()), inplace=True)
test['Credit_History'].fillna(value=float(test['Credit_History'].mode()), inplace=True)



# ----- MODEL EVALUATION

# Model Evaluation results -> name of best algorithm (alg), its accuracy score (alg_acc_scr) and best K for KNN algorithm (maxK)
alg, alg_acc_scr, maxK = ModelEvaluation(train)
print("\n\talg", alg, alg_acc_scr, maxK, '\n\n')

X = train.drop('Loan_Status', axis=1)
y = train['Loan_Status']

# Choosing the best algorithm after model evaluation (line 391)
if alg == 'Decision Tree':
    dtree = DecisionTreeClassifier()
    dtree.fit(X, y)
    predictions = dtree.predict(test)
    test['Loan_Status'] = predictions

if alg == 'Random Forest':
    rfc = RandomForestClassifier(n_estimators=100)
    rfc.fit(X, y)
    predictions = rfc.predict(test)
    test['Loan_Status'] = predictions

if alg == 'KNN':
    knn = KNeighborsClassifier(n_neighbors=maxK)
    knn.fit(X, y)
    predictions = knn.predict(test)
    test['Loan_Status'] = predictions

if alg == 'SVM':
    model = SVC()
    model.fit(X, y)
    predictions = model.predict(test)
    test['Loan_Status'] = predictions

if alg == 'Logistic Regression':
    logmodel = LogisticRegression(solver='lbfgs')
    logmodel.fit(X, y)
    predictions = logmodel.predict(test)
    test['Loan_Status'] = predictions

# Creating a Dataframe that stores 'Loan_ID' and 'Loan_Status' - the results file must have only these two columns
res_df = pd.DataFrame()
res_df['Loan_ID'] = loanID_df['Loan_ID']
res_df['Loan_Status'] = test['Loan_Status']
res_df['Loan_Status'] = res_df['Loan_Status'].apply(lambda row: 'Y' if row == 1 else 'N')

# Exporting the predictions to csv file
export_csv = res_df.to_csv(inout_path + "\\LoanPredictionIII_results2.csv", index=None, header=True)



