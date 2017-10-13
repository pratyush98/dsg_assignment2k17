# dsg_assignment2k17
It includes analyzing the mushroom dataset and predicting its edibility(please download the ipnb notebook to view it)

PROBLEM STATEMENT:

we need to find the edibility of mushroom using some features like radius,weight,cap size,cap colour etc

STEPS USED:

1-DATA IMPUTATIONS(Removing null and zeroes/removing zeroes by predicting it using descison tree

2-Use label encoder to encode unique data

3-using heatmaps and pairplots to see any features that are having lesser covariance with other features.

4-using PCA to do dimensional reduction.draw the cummulative explained variance ratio and use the no.
 of features derived from PCA that have a significant explained variance ratio.

5-using all types of classification(LogisticRegression(),DecisionTreeClassifier(),RandomForestClassifier(),SVC(),
AdaBoostClassifier(),SGDClassifier())on the PCA model of n=20 components,training the models and use the best 
model of them.

6-use the train data use the same PCA weights for each attribute to get same 20 attributes as the training data 
set fit it in that model and then fit the data in Random tree classifier to get the best fit.
7-finally we take the 
test and train datasets combine them and divide it in train and test datasets and again fit those in all calssification 
models and find accuracy.
