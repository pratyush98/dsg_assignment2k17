
# coding: utf-8

# In[1]:


import numpy as np
import sklearn
import pandas as pd
df1=pd.read_csv("C:/Users/Satyabrat Behera/Downloads/mushroom_train.csv")
df1.head()


# In[2]:


df1.isnull().sum()#no null should be present


# In[3]:


df1['class'].unique()#to see no. of unique elements


# In[4]:


from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
for col in df1.columns:
    df1[col] = labelencoder.fit_transform(df1[col])
df1.head()#unique


# In[5]:


for feature,col_data in df1.iteritems():
    if col_data.dtype == object:
        print("{} has {}".format(feature,col_data.unique()))


# In[6]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[7]:


X = df1.iloc[:,1:24]  # all rows, all the features and no labels
y = df1.iloc[:, 0]  # all rows, label only
X.drop('veil-type',axis=1)
df1=df1.drop('veil-type',axis=1)
X.shape


# In[8]:


corr_mat=df1.corr()


# In[9]:


corr_mat#finding correlation of each element with the other one.


# In[10]:


get_ipython().magic('matplotlib inline')
corr_mat1=np.array(corr_mat)
corr_mat1


# In[11]:


f, ax = plt.subplots(figsize=(17,17))
ax=sns.heatmap(corr_mat,annot=True,cmap='seismic_r',ax=ax,linewidths=.5)


# In[12]:


f, ax = plt.subplots(figsize=(17,17))
ax=sns.countplot(x='radius',data=df1,ax=ax)


# In[13]:


df1['weight']


# In[14]:


df1['radius']


# In[15]:


ctr=0
for i in df1['weight'].values:
    if i==0:
        ctr=ctr+1
ctr


# In[16]:


ctr=0
for i in df1['radius'].values:
    if i==0:
        ctr=ctr+1
ctr


# In[17]:


#imputations on weight
xtest=df1.drop('weight',axis=1)
ytest=df1['weight']
xpred=df1.drop('weight',axis=1)
ypred=df1['weight']
ctr=0
c=0
for i in df1['weight']:
    if i==0: 
        xpred=xpred.drop(ctr)
        ypred=ypred.drop(ctr)
    else:
        xtest=xtest.drop(ctr)
        ytest=ytest.drop(ctr)
    ctr=ctr+1
ytest


# In[18]:


from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
model1=DecisionTreeClassifier().fit(xpred,ypred)
res=model1.predict(xtest)


# In[19]:


res


# In[20]:


res.shape


# In[21]:


ctr=0
ctr1=0
for i in df1['weight']:
    if i==0: 
        df1['weight'][ctr1]=res[ctr]
        ctr=ctr+1
    ctr1=ctr1+1


# In[22]:


df1


# In[23]:


#imputations on weight
xtest=df1.drop('radius',axis=1)
ytest=df1['radius']
xpred=df1.drop('radius',axis=1)
ypred=df1['radius']
ctr=0
c=0
for i in df1['radius']:
    if i==0: 
        xpred=xpred.drop(ctr)
        ypred=ypred.drop(ctr)
    else:
        xtest=xtest.drop(ctr)
        ytest=ytest.drop(ctr)
    ctr=ctr+1
ytest


# In[24]:


model2=DecisionTreeClassifier().fit(xpred,ypred)
res=model2.predict(xtest)


# In[25]:


res


# In[26]:


ctr=0
ctr1=0
for i in df1['radius']:
    if i==0: 
        df1['radius'][ctr1]=res[ctr]
        ctr=ctr+1
    ctr1=ctr1+1


# In[27]:


df1['radius']


# In[28]:


corr_mat2=df1.corr()
corr_mat2.shape


# In[29]:


f, ax = plt.subplots(figsize=(17,17))
ax=sns.heatmap(corr_mat2,annot=True,cmap='seismic_r',ax=ax,linewidths=.5)


# In[30]:



ax=sns.jointplot(x='cap-shape',y='cap-color',data=df1)


# In[31]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X=scaler.fit_transform(X)
df2=scaler.fit_transform(df1)
df2=pd.DataFrame(df2)
X.shape


# In[32]:


ax=sns.jointplot(x='weight',y='radius',data=df1)


# In[33]:


f, ax = plt.subplots(figsize=(10,10))
ax=sns.regplot(x='weight',y='radius',data=df1,ax=ax)


# In[34]:


g = sns.pairplot(df1,hue="class",palette="muted",size=5,
    vars=["weight", "radius"],kind='reg',markers=['o','x'])


# In[35]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)


# In[36]:


pca.fit(X)


# In[37]:


var1= pca.explained_variance_ratio_


# In[38]:


varper2=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)


# In[39]:


varper2


# In[40]:


proj=(df2.drop(0,axis=1)).values
pca = PCA(n_components=2)
x = pca.fit_transform(proj)
plt.figure(figsize = (15,15))
plt.scatter(x[:,0],x[:,1])
plt.show()
x.shape


# In[41]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=5)
X_clustered = kmeans.fit_predict(proj)

LABEL_COLOR_MAP = {0 : 'g',
                   1 : 'y',
                   2 : 'r',
                   3 : 'b',
                   4 : 'cyan',
                   5 : 'brown'
                  }

label_color = [LABEL_COLOR_MAP[l] for l in X_clustered]
plt.figure(figsize = (15,7))
plt.scatter(x[:,0],x[:,1], c= label_color)
plt.show()


# In[42]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, random_state=5)
X_clustered = kmeans.fit_predict(proj)

LABEL_COLOR_MAP = {0 : 'g',
                   1 : 'y',
                   2 : 'r',
                   3 : 'b',
                   4 : 'cyan',
                   5 : 'brown'
                  }

label_color = [LABEL_COLOR_MAP[l] for l in X_clustered]
plt.figure(figsize = (15,7))
plt.scatter(x[:,0],x[:,1], c= label_color)
plt.show()


# In[43]:


models ={'Logistic Regression':LogisticRegression(),'Decision Tree Classifier':DecisionTreeClassifier(),
         'Random Forest':RandomForestClassifier(),'Support Vector Machines':SVC(),'AdaBoost Classifier':AdaBoostClassifier(),
         'Stochastic Gradient Descent':SGDClassifier()}


# In[44]:


pca1=PCA(n_components=22)


# In[45]:


pca1.fit(X)#we only fit the model because the model gives us the preffered weights for eacgh of attributes 
#for the train data so that the our principal components have the same orientation in the model while predicting


# In[46]:


var11= pca1.explained_variance_ratio_
varper21=np.cumsum(np.round(pca1.explained_variance_ratio_, decimals=4)*100)


# In[47]:


varper21


# In[48]:


plt.plot(varper21)


# In[49]:


#we see that after taking 20 principal components we get a saturation of cumulative variance 
#so we take 20 components to build our model
X1=pca1.fit_transform(X)
print(X1.shape)


# In[50]:


from sklearn.model_selection import train_test_split # for splitting the data

# Normally data is split into 70,15,15 % for training , validation and testing respectivlely

X_train, X_valid, Y_train, Y_valid = train_test_split(X1, y , test_size=0.30, random_state=42)
X_train.shape


# In[51]:


scores_precision=[]
scores_acc=[]
scores_recall=[]
scores_f1score=[]
names=[]
from sklearn import metrics
for name,model in models.items():
    model.fit(X_train,Y_train)
    names.append(name)
    precision=metrics.precision_score(Y_valid,model.predict(X_valid))
    scores_precision.append(precision)
    acc=metrics.accuracy_score(Y_valid,model.predict(X_valid))
    scores_acc.append(acc)
    recall=metrics.recall_score(Y_valid,model.predict(X_valid))
    scores_recall.append(recall)
    f1score=metrics.f1_score(Y_valid,model.predict(X_valid))
    scores_f1score.append(f1score)
   
    dataframe = pd.DataFrame({'Models':names,'Precision':scores_precision,'F1-score':scores_f1score,
                              'Accuracy':scores_acc,'Recall':scores_recall})
    
cols = list(dataframe)
# move the column to head of list using index, pop and insert
cols.insert(0, cols.pop(cols.index('Models')))
dataframe = dataframe.ix[:, cols]
dataframe


# In[52]:


#random forest classifier and svm are best suited for classification
dframetest=pd.read_csv("C:/Users/Satyabrat Behera/Downloads/mushroom_test.csv")
dframetest


# In[53]:



labelencoder=LabelEncoder()
for col in dframetest.columns:
    dframetest[col] = labelencoder.fit_transform(dframetest[col])
dframetest.head()


# In[54]:


df1.shape


# In[55]:


X.shape


# In[56]:


dframetest=dframetest.drop('veil-type',axis=1)


# In[57]:


dframetest.head()


# In[58]:


#imputations on weight
xtest=dframetest.drop('weight',axis=1)
ytest=dframetest['weight']
xpred=dframetest.drop('weight',axis=1)
ypred=dframetest['weight']
ctr=0
c=0
for i in dframetest['weight']:
    if i==0: 
        xpred=xpred.drop(ctr)
        ypred=ypred.drop(ctr)
    else:
        xtest=xtest.drop(ctr)
        ytest=ytest.drop(ctr)
    ctr=ctr+1
ytest


# In[59]:


model3=DecisionTreeClassifier().fit(xpred,ypred)
res=model3.predict(xtest)


# In[60]:


ctr=0
ctr1=0
for i in dframetest['weight']:
    if i==0: 
        dframetest['weight'][ctr1]=res[ctr]
        ctr=ctr+1
    ctr1=ctr1+1


# In[61]:


dframetest.head()


# In[62]:


xtest=dframetest.drop('radius',axis=1)
ytest=dframetest['radius']
xpred=dframetest.drop('radius',axis=1)
ypred=dframetest['radius']
ctr=0
c=0
for i in dframetest['radius']:
    if i==0: 
        xpred=xpred.drop(ctr)
        ypred=ypred.drop(ctr)
    else:
        xtest=xtest.drop(ctr)
        ytest=ytest.drop(ctr)
    ctr=ctr+1
ytest


# In[63]:


model4=DecisionTreeClassifier().fit(xpred,ypred)
res=model4.predict(xtest)


# In[64]:


ctr=0
ctr1=0
for i in dframetest['radius']:
    if i==0: 
        dframetest['radius'][ctr1]=res[ctr]
        ctr=ctr+1
    ctr1=ctr1+1


# In[65]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
Xsc=scaler.fit_transform(dframetest)


# In[66]:


Xsc


# In[67]:


dframetest1=scaler.fit_transform(dframetest)
dframetest1=pd.DataFrame(dframetest1)
dframetest1.head()


# In[68]:


X4=pca1.fit_transform(Xsc)
X5=pca.fit_transform(Xsc)


# In[69]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=5)
X_clustered1 = kmeans.fit_predict(np.array(dframetest1))

LABEL_COLOR_MAP = {0 : 'g',
                   1 : 'y',
                   2 : 'r',
                   3 : 'b',
                   4 : 'cyan',
                   5 : 'brown'
                  }

label_color = [LABEL_COLOR_MAP[l] for l in X_clustered1]
plt.figure(figsize = (15,7))
plt.scatter(X5[:,0],X5[:,1], c= label_color)
plt.show()


# In[70]:


yresult=models['Random Forest'].predict(X4)


# In[71]:


yresult=pd.DataFrame(yresult,columns=['class'])


# In[72]:


col2=dframetest.columns


# In[73]:


col2


# In[74]:


#move class to first column
dframetest.shape


# In[75]:


#concat them together
dresult=yresult.join(dframetest)
dresult.head()


# In[76]:


df1.head()


# In[77]:


#to calculate accuracy of our prediction we need to contactinate and predict the accuracy
combine=df1.append(dresult,ignore_index=True)


# In[78]:


combine.shape


# In[79]:


yfinal=combine['class']
xfinal=combine.drop('class',axis=1)


# In[80]:


xfinal.head()


# In[85]:


dresult.to_csv('finalresult.csv', sep='\t', encoding='utf-8')


# In[88]:


#scale all data and use it to train another model using same pca weights and finding the accuracy
scaler = StandardScaler()
xfinal=scaler.fit_transform(xfinal)
xfinal=pca1.fit_transform(xfinal)
X_trainfinal, X_validfinal, Y_trainfinal, Y_validfinal = train_test_split(xfinal, yfinal , test_size=0.30, random_state=42)


# In[89]:


scores_precisionf=[]
scores_accf=[]
scores_recallf=[]
scores_f1scoref=[]
namesf=[]
for name,modelfinal in models.items():
    modelfinal.fit(X_trainfinal,Y_trainfinal)
    namesf.append(name)
    precisionf=metrics.precision_score(Y_validfinal,modelfinal.predict(X_validfinal))
    scores_precisionf.append(precisionf)
    accf=metrics.accuracy_score(Y_validfinal,modelfinal.predict(X_validfinal))
    scores_accf.append(accf)
    recallf=metrics.recall_score(Y_validfinal,modelfinal.predict(X_validfinal))
    scores_recallf.append(recallf)
    f1scoref=metrics.f1_score(Y_validfinal,modelfinal.predict(X_validfinal))
    scores_f1scoref.append(f1scoref)
dataframeaccrfinal = pd.DataFrame({'Models':namesf,'Precision':scores_precisionf,'F1-score':scores_f1scoref,
                              'Accuracy':scores_accf,'Recall':scores_recallf})
colsf = list(dataframeaccrfinal)
# move the column to head of list using index, pop and insert
colsf.insert(0, colsf.pop(colsf.index('Models')))
dataframeaccrfinal = dataframeaccrfinal.ix[:, colsf]
dataframeaccrfinal


# In[83]:


#so we got a maximum accuracy of 95.406% using random forest classification..

