#!/usr/bin/env python
# coding: utf-8

# # Predicting heart disease using Machine Learning (ML)
# This notebook looks into using various Python based ML and data science libraries in an attempt to build a ML learning model capable of predicting whether or not someone has heart disease based on their medical attributes.
# 
# The approach that will be taken:
# 1. Problem definition
# 2. Data
# 3. Evaluation
# 4. Features
# 5. Modelling
# 6. Experimentation
# 
# ## 1. Problem Definition
# 
# > Given clinical parameter about a patient, can we predict whether or not they have heart disease?
# 
# ## 2. Data
# 
# The original data came from the Cleavland from the UCI Machine Learning Repository.
# 
# ## 3. Evaluation
# 
# > If we can reach 95% accuracy at predicting whether or not a patient has heart disease during the proof of concept, we'll pursue the project.
# 
# ## 4. Features
# 
# **Create data dictionary**
# 
# 1. age - age in years
# 2. sex - (1 = male; 0 = female)
# 3. cp - chest pain type
#     * 0: Typical angina: chest pain related decrease blood supply to the heart
#     * 1: Atypical angina: chest pain not related to heart
#     * 2: Non-anginal pain: typically esophageal spasms (non heart related)
#     * 3: Asymptomatic: chest pain not showing signs of disease
# 4. trestbps - resting blood pressure (in mm Hg on admission to the hospital)
#     anything above 130-140 is typically cause for concern
# 5. chol - serum cholestoral in mg/dl
#     serum = LDL + HDL + .2 * triglycerides
#     above 200 is cause for concern
# 6. fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
#     '>126' mg/dL signals diabetes
# 7. restecg - resting electrocardiographic results
#     * 0: Nothing to note
#     * 1: ST-T Wave abnormality
#     can range from mild symptoms to severe problems
#     signals non-normal heart beat
#     * 2: Possible or definite left ventricular hypertrophy
#     Enlarged heart's main pumping chamber
# 8. thalach - maximum heart rate achieved
# 9. exang - exercise induced angina (1 = yes; 0 = no)
# 10. oldpeak - ST depression induced by exercise relative to rest
#     looks at stress of heart during excercise
#     unhealthy heart will stress more
# 11. slope - the slope of the peak exercise ST segment
#     * 0: Upsloping: better heart rate with excercise (uncommon)
#     * 1: Flatsloping: minimal change (typical healthy heart)
#     * 2: Downslopins: signs of unhealthy heart
# 12. ca - number of major vessels (0-3) colored by flourosopy
#     colored vessel means the doctor can see the blood passing through
#     the more blood movement the better (no clots)
# 13. thal - thalium stress result
#     1,
#     * 3: normal
#     * 6: fixed defect: used to be defect but ok now
#     * 7: reversable defect: no proper blood movement when excercising
# 14. target - have disease or not (1=yes, 0=no) (= the predicted attribute)
# 
# ## 5. Modeling
# 
# 
# ## 6. Experimentation

# ## Preparing the tools
# 
# We're going to use pandas, matplotlib and NumPy for data analysis and manipulation.

# In[5]:


# Import all the tools we need

# Regular EDA (exploratory data analysis) and plotting libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# we want our plots to appear inside the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Model from sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Model Evaluation
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import plot_roc_curve


# ## Load Data

# In[6]:


df = pd.read_csv("11.2 heart-disease.csv")
df.shape # (rows, columns)


# ## Data Exploration (EDA)
# 
# The goal here is to find out more about the data and become an expert on the dataset. This is to avoid overfitting and inderfitting.
# 
# 1. What questions are you trying to solve?
# 2. What kind of data do we have and how do we treat different types?
# 3. What's missing from the dataand how do you deal with it?
# 4. Where are the outliers and why should you care about them?
# 5. How can you add, change or remove features to get more out of your data?

# In[7]:


df.head()


# In[8]:


df.tail()


# In[9]:


# How many of each class there are
df.target.value_counts()


# In[10]:


df.target.value_counts().plot(kind="bar", color=["red", "blue"]);


# In[11]:


df.info()


# In[12]:


# Checking for missing values
df.isna().sum()


# In[13]:


df.describe()


# ### Heart Disease Frequency accarding to Sex

# In[14]:


df.sex.value_counts()


# In[15]:


# Compare target column with sex column
pd.crosstab(df.target, df.sex)


# In[16]:


# create a plot of crosstab
pd.crosstab(df.target, df.sex).plot(kind="bar",
                                    figsize=(10, 6),
                                    color=["red", "blue"])

plt.title("Heart Disease Frequency by Sex")
plt.xlabel("0 = No Disease, 1 = Heart Disease")
plt.ylabel("Ammount")
plt.legend(["Female", "Male"])
plt.xticks(rotation=0);


# In[17]:


df.head()


# In[18]:


pd.crosstab(df.target, df.chol)


# In[19]:


plt.figure(figsize=(10,6))

plt.scatter(df.age[df.target==1],
            df.oldpeak[df.target==1],
            c="red")
plt.scatter(df.age[df.target==0],
            df.oldpeak[df.target==0],
            c="blue")
plt.title("Heart Disease in function of Age and Level of exercise")
plt.xlabel("Age")
plt.ylabel("Level of Exercise")
plt.legend(["Disease", "No Disease"]);


# ### Age vs. Max Heart Rate for Heart disease

# In[20]:


# Create anothe figure
plt.figure(figsize=(10, 6))

# Scatter with positive examples
plt.scatter(df.age[df.target==1],
            df.thalach[df.target==1],
            c="red")

# scatter with negative examples
plt.scatter(df.age[df.target==0],
            df.thalach[df.target==0],
            c="blue")

# Some helpful info
plt.title("Heart Disease in function of Age and Max Heart Rate")
plt.xlabel("Age")
plt.ylabel("Max Heart Rate")
plt.legend(["Disease", "No Disease"]);


# In[21]:


# Check the distribution of the age column with a histogram
df.age.plot.hist();


# ### Heart Disesease Frequency per chest pain type
# 
# 3. cp - chest pain type
#     * 0: Typical angina: chest pain related decrease blood supply to the heart
#     * 1: Atypical angina: chest pain not related to heart
#     * 2: Non-anginal pain: typically esophageal spasms (non heart related)
#     * 3: Asymptomatic: chest pain not showing signs of disease

# In[22]:


pd.crosstab(df.cp, df.target)


# In[23]:


# Make the crosstab more visual
pd.crosstab(df.cp, df.target).plot(kind="bar",
                                   figsize=(10, 6),
                                   color=["red", "blue"])

# Add communication
plt.title("Heart Disease Frequency Per Chest Pain Type")
plt.xlabel("Chest Pain Type")
plt.ylabel("Ammount")
plt.legend(["No Disease", "Disease"])
plt.xticks(rotation=0);


# In[24]:


# Make a correlation matrix
df.corr()


# In[25]:


# make the correlation matrix pretty
corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(15, 10))
ax = sns.heatmap(corr_matrix,
                 annot=True,
                 linewidths=0.5,
                 fmt=".2f",
                 cmap="YlGnBu");


# ## 5.0 Modelling

# In[26]:


df.head()


# In[27]:


# split data into X and y
X = df.drop("target", axis=1)

y = df["target"]


# Three different models will be used: RandomForrestClassifier, KNeighborsClassifier, and LogisticRegression

# In[28]:


# RandomForestClassifier
# split data into train and test sets
np.random.seed(42)

# split into train and test set
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2)
# Use RandomForrestClassifier
model = RandomForestClassifier().fit(X_train, y_train)


# In[29]:


# Put models in a dictionary
models = {"Logistic Regression": LogisticRegression(),
          "KNN": KNeighborsClassifier(),
          "Random Forest": RandomForestClassifier()}

# Create a function to fit and score models
def fit_and_score(models, X_train, X_test, y_train, y_test):
    """""
    Fits and evaluates given machine learning models.
    models: a dictionary of different sklearn ml models
    X_train: training data (no labels)
    X_test: testing data (no labels)
    y_train: training labels
    y_test: test labels
    """
    # ste random seed
    np.random.seed(42)
    # make a dictionary to keep model scores
    model_scores = {}
    # loop through models
    for name, model in models.items():
        # Fit the model to the data
        model.fit(X_train, y_train)
        # evaluate the model and append its score to model_scores
        model_scores[name] = model.score(X_test, y_test)
    return model_scores


# In[30]:


model_scores = fit_and_score(models=models,
                             X_train=X_train,
                             y_train=y_train,
                             X_test=X_test,
                             y_test=y_test)

model_scores


# ## Model Comparison

# In[27]:


model_compare = pd.DataFrame(model_scores, index=["accuracy"])
model_compare.T.plot.bar();


# A models first predictions aren't what we should base our next steps off. What should we do?
# 
# Let's look at the following:
# * Hyperparameter tuning
# * Feature importance
# * Confusion matrix
# * Cross-validation
# * Precision
# * Recall
# * F1 score
# * Classification report
# * ROC curve
# * Area under the curve (AUC)
# 
# ### Hyperparameter tuning (by hand)

# In[28]:


# Tune KNN

train_scores = []
test_scores = []

# Create a list of values of n_neighbors
neighbors = range(1, 21)

# setup KNN instance
knn = KNeighborsClassifier()

# Loop through different n_neighbors
for i in neighbors:
    knn.set_params(n_neighbors=i).fit(X_train, y_train)
    
    # update training scores list
    train_scores.append(knn.score(X_train, y_train))
    
    # update the test scores list
    test_scores.append(knn.score(X_test, y_test))


# In[29]:


train_scores


# In[30]:


test_scores


# In[31]:


plt.plot(neighbors, train_scores, label="Train score")
plt.plot(neighbors, test_scores, label="Test score")
plt.xlabel("Number of neighbors")
plt.ylabel("Model score")
plt.legend()

print(f"Max KNN score on the test data: {max(test_scores)*100: .2f}%")


# ## Hyperparameter tuning with RandomizedSearchCV
# 
# Wh at is going to be tuned:
# * LogisticRegression()
# * RandomForestClassifier()
# 
# With RandomizedSearchCV

# In[32]:


# Create a hyperameter grid for LogisticRegression
log_reg_grid = {"C": np.logspace(-4, 4, 20),
                "solver": ["liblinear"]}

# create a hyperparameter grid for RandonForestClassifier
rf_grid = {"n_estimators": np.arange(10, 1000, 50),
           "max_depth": [None, 3, 5, 10],
           "min_samples_split": np.arange(2, 20, 2),
           "min_samples_leaf": np.arange(1, 20, 2)}


# Use grids to tune using RandomizedSearchCV

# In[33]:


# Tune LogisticRegression

np.random.seed(42)

# setup random hyperparameter search for model
rs_log_reg = RandomizedSearchCV(LogisticRegression(),
                                param_distributions=log_reg_grid,
                                cv=5,
                                n_iter=20,
                                verbose=True)

#fit random hyperparmeter search model
rs_log_reg.fit(X_train, y_train)


# In[34]:


rs_log_reg.best_params_


# In[35]:


rs_log_reg.score(X_test, y_test)


# Now to tune RandomForestClassifier

# In[ ]:


# tune RandomForestClassifier

np.random.seed(42)

rs_rf = RandomizedSearchCV(RandomForestClassifier(),
                             param_distributions=rf_grid,
                             cv=5,
                             n_iter=20,
                             verbose=True)

rs_rf.fit(X_train, y_train)


# In[ ]:


rs_rf.score(X_test, y_test)


# ## Hyperparameter Tuning using GridSearchCV
# 
# Since our LogisticRegression model provided the best score so far, now I'll try with GridSearchCV

# In[50]:


# Different hyperparameters for log_reg model
log_reg_grid = {"C": np.logspace(-4, 4, 30),
                "solver": ["liblinear"]}

# setup grid hyperparameter search
gs_log_reg = GridSearchCV(LogisticRegression(),
                          param_grid=log_reg_grid,
                          cv=5,
                          verbose=True)

# fitgrid hyperparameter search model
gs_log_reg.fit(X_train, y_train)


# In[51]:


# Check the best
gs_log_reg.best_params_


# In[52]:


# evaluate the model
gs_log_reg.score(X_test, y_test)


# ## Evaluate tuned ml classifier
# 
# * ROC curve and AUC score
# * Confusion matrix
# * Precision
# * Recall
# * F1 score

# In[55]:


y_preds = gs_log_reg.predict(X_test)
y_preds


# In[56]:


# Plot ROC curve and calculate AUC metric
plot_roc_curve(gs_log_reg, X_test, y_test)


# In[65]:


# confusion matrix
print(confusion_matrix(y_test, y_preds))


# In[72]:


sns.set(font_scale=1.5)

def plot_conf_mat(y_test, y_preds):
    """"
    PLots a nice looking confusion matrix
    """
    fig, ax = plt.subplots(figsize=(3, 3))
    ax = sns.heatmap(confusion_matrix(y_test, y_preds),
                     annot=True,
                     cbar=False)
    plt.xlabel("True label")
    plt.ylabel("Predicted label")
    
plot_conf_mat(y_test, y_preds)


# Get a classificaton report as well as cross-validated precision, recall and f1 score

# In[73]:


print(classification_report(y_test, y_preds))


# ### Calculate elvaluation metrics using cross-validaton

# In[74]:


# Check best hyperparameters
gs_log_reg.best_params_


# In[75]:


clf = LogisticRegression(C=0.20433597178569418,
                         solver="liblinear")


# In[95]:


# Cross-validated accuracy
cv_acc = cross_val_score(clf,
                         X,
                         y,
                         cv=5,
                         scoring="accuracy")
cv_acc = np.mean(cv_acc)
cv_acc


# In[94]:


# cross-validated precision
cv_precision = cross_val_score(clf,
                         X,
                         y,
                         cv=5,
                         scoring="precision")
cv_precision = np.mean(cv_precision)
cv_precision


# In[93]:


# cross-validated recall
cv_recall = cross_val_score(clf,
                         X,
                         y,
                         cv=5,
                         scoring="recall")
cv_recall = np.mean(cv_recall)
cv_recall


# In[92]:


# cross-validated f1 score
cv_f1 = cross_val_score(clf,
                         X,
                         y,
                         cv=5,
                         scoring="f1")
cv_f1 = np.mean(cv_f1)
cv_f1


# In[97]:


# visualize cross-validated metrics
cv_metrics = pd.DataFrame({"Accuracy": cv_acc,
                           "Precision": cv_precision,
                           "Recall": cv_recall,
                           "f1": cv_f1},
                           index=[0])

cv_metrics.T.plot.bar(title="Cross-validated classification metrics",
                      legend=False);


# ### Feature Importance
# 
# Feature importance is a way of asking which features contributed most to the outcomes of model
# 
# Finding feature importance is different for each ml model.
# 
# Find the feature importance for our model.

# In[99]:


# fit an instance of LogisticREgression
clf = LogisticRegression(C=0.20433597178569418,
                         solver="liblinear")

clf.fit(X_train, y_train);


# In[100]:


# check coef_
clf.coef_


# In[102]:


# Match coef's of features to columns
feature_dict = dict(zip(df.columns, list(clf.coef_[0])))
feature_dict


# In[103]:


# visualize feature importance
feature_df = pd.DataFrame(feature_dict, index=[0])
feature_df.T.plot.bar(title="Feature Importance", legend=False);


# In[105]:


pd.crosstab(df["slope"], df["target"])


# ### Conclusion
# 
# I haven't hit my evalutation metric.
# 
# * Could you collect more data?
# * Could you try a better model? Like CatBoost or XGBoost?
# * Could you improve the current models?

# In[1]:


conda install -c anaconda py-xgboost


# In[38]:


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


# ### Using XGBCLassifier

# In[39]:


# split data into train and test sets
np.random.seed(42)

# split into train and test set
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2)
# Use XGBClassifier
model = XGBClassifier().fit(X_train, y_train)

#predictions
y_preds = model.predict(X_test)
predictions = [round(value) for value in y_preds]

#evaluate preds
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[ ]:




