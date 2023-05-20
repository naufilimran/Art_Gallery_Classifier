#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pingouin as pg
import scipy.stats as stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
#here we import all the libraries that we would be using for the whole project.


# In[972]:


art = pd.read_csv("art.csv")
art
#The art file imported as a pandas dataframe.


# In[973]:


row_info = art.loc[26]
row_info


# In[974]:


data_set = pd.read_csv("datacapstone.csv",header=None)
data_set
#this is our main data file where i have used this file to make smalled subsets for each questions.


# In[975]:


row_info = data_set.loc[26]
row_info


# In[976]:


#Q1 STARTS HERE
data_q1 = data_set.iloc[:, 0:91]
data_q1_transposed = data_q1.T
data_q1_transposed
# in the data_q1_transposed 1 row of mine is a movie and the 300 columns represent the the user ratings of 300 individuals.
data_q1_transposed = pd.merge(data_q1_transposed, art['Source (1 = classical, 2 = modern, 3 = nonhuman)'], left_index=True, right_index=True)
data_q1_transposed.head(5)


# In[977]:


row_data = data_q1_transposed.iloc[:,:300]
row_mean= row_data.mean(axis=1)
row_mean.hist()
plt.hist(row_mean)
plt.xlabel('Mean Ratings')
plt.ylabel('Frequency')
plt.title('Plotting means to check for normality')
plt.show()


# In[978]:


classical_data = data_q1_transposed[data_q1_transposed['Source (1 = classical, 2 = modern, 3 = nonhuman)'] == 1]
modern_data = data_q1_transposed[data_q1_transposed['Source (1 = classical, 2 = modern, 3 = nonhuman)'] == 2]
classical_data = classical_data.iloc[:,:300]
modern_data = modern_data.iloc[:,:300]
classical_data_mean = classical_data.mean().mean()
modern_data_mean = modern_data.mean().mean()


# In[979]:


fig, ax = plt.subplots(figsize=(5, 3))
colors = ['#FFC300', '#DAF7A6']
ax.bar(['Classical', 'Modern'], [classical_data_mean, modern_data_mean],color=colors)

# Add labels and title
ax.set_xlabel('Art Category')
ax.set_ylabel('Average Rating')
ax.set_title('Classical Art is Preferred over Modern Art')


# In[980]:



from scipy.stats import wilcoxon
wilcoxon_stats, p_value = wilcoxon(classical_data.values.flatten(), modern_data.values.flatten())
print("p-value:", p_value)


# In[981]:


#Q2 STARTS HERE
non_human = art[['computerOrAnimal (0 = human, 1 = computer, 2 = animal)']]
non_human =  non_human[non_human['computerOrAnimal (0 = human, 1 = computer, 2 = animal)'] >= 1]

non_human_data = data_set.iloc[:, 70:91]


# In[982]:


non_human_mean = non_human_data.mean().mean()


# In[983]:


modern_data_mean = modern_data.T.mean().mean()
fig, ax = plt.subplots(figsize=(5, 3))
colors = ['#FFC300', '#DAF7A6']
ax.bar(['Non Human', 'Modern'], [non_human_mean, modern_data_mean],color=colors)

# Add labels and title
ax.set_xlabel('Art Category')
ax.set_ylabel('Average Rating')
ax.set_title('Modern Art is Preferred over Non human art')


# In[984]:



from scipy.stats import mannwhitneyu
statistic, pvalue = mannwhitneyu(modern_data.T.values.flatten(),non_human_data.values.flatten())
print("the p value for question 2 is,", pvalue)


# In[985]:


#Q3 STARTS HERE
data_q3= data_set.iloc[:, list(range(91)) + [216]]
# Remove rows with NaN values
data_q3 = data_q3.dropna()

# Remove rows in column 216 that have a value of 3
data_q3 = data_q3[data_q3.iloc[:, -1] != 3]
data_q3


# In[986]:


male_rating = data_q3[data_q3.iloc[:, -1] == 1]
male_rating_1 = male_rating.iloc[:95,:91]
male_rating_1


# In[987]:


female_rating = data_q3[data_q3.iloc[:, -1] == 2]
female_rating_1 = female_rating.iloc[:179, :91]
female_rating_1


# In[ ]:





# In[988]:


plt.hist([male_rating_1.mean(), female_rating_1.mean()], bins=10, label=['Male', 'Female'])
plt.xlabel('mean ratings')
plt.ylabel('Frequency')
plt.title('Art preference ratings')
plt.legend()
plt.show()


# In[989]:


stat, p = mannwhitneyu(female_rating_1.values.flatten(),male_rating_1.values.flatten())
print("Mann-Whitney U test statistic:", stat)
print("p-value:", p)


# In[990]:


#Q4 STARTS HERE
data_q4 = data_set.iloc[:, list(range(91)) + [218]]


# In[991]:


data_q4 = data_q4.dropna()
data_q4


# In[992]:


zero_art_knowledge =  data_q4[data_q4.iloc[:, -1] == 0]
zero_art_knowledge = zero_art_knowledge.iloc[:92,:91]
zero_art_knowledge


# In[993]:


art_knowledge = data_q4[data_q4.iloc[:, -1] != 0]
art_knowledge = art_knowledge.iloc[:188,:91]
art_knowledge


# In[994]:


stat, p = mannwhitneyu(art_knowledge.values.flatten(),zero_art_knowledge.values.flatten())
print("Mann-Whitney U test statistic:", stat)
print("p-value:", p)


# In[995]:


plt.hist([art_knowledge.mean(axis=1), zero_art_knowledge.mean(axis=1)], bins=10, label=['Art Knoweldge', 'Zero Art Knowledge'])
plt.xlabel('mean ratings')
plt.ylabel('Frequency')
plt.title('Art preference ratings')
plt.legend()
plt.show()


# In[996]:


#Q5
data_art_q5 = data_set.iloc[:,0:91]
data_art_q5 = data_art_q5.dropna().mean(axis=1)
data_art_q5


# In[997]:


data_energy_q5 = data_set.iloc[:,91:182]
data_energy_q5 = data_energy_q5.dropna()
data_energy_q5 = data_energy_q5.mean(axis=1)
data_energy_q5


# In[998]:


reshaped_data_art = data_art_q5.to_numpy().reshape(300, -1)


# In[999]:


row_data_energy = data_energy_q5.to_numpy().reshape(300, -1)


# In[1000]:


plt.scatter(row_data_energy, reshaped_data_art)


# In[ ]:





# In[1058]:


X_train_q5, X_test_q5, y_train_q5, y_test_q5 = train_test_split(row_data_energy, reshaped_data_art, test_size=0.3, random_state=16790632)
model_q5 = LinearRegression()
model_q5.fit(X_train_q5, y_train_q5)
y_pred_q5 = model_q5.predict(X_test_q5)
r_squared_q5 = model_q5.score(X_test_q5, y_test_q5)
print("R-squared score:", r_squared_q5)

rmse_q5 = np.sqrt(mean_squared_error(y_test_q5, y_pred_q5))
print("RMSE:", rmse_q5)


# In[1059]:


results_data_q5 = pd.DataFrame({'Actual': y_test_q5.flatten(), 'Predicted': y_pred_q5.flatten()})
results_data_q5.head(3)


# In[1060]:


graph_q5=results_data_q5.head(15)
graph_q5.plot(kind='bar')
plt.title('Predicting art prefence ratings from energy ratings', fontsize=15)
plt.ylabel('Mean rating for an art',fontsize=15)
plt.xlabel('Actual and predicted ratings',fontsize=10)


# In[1061]:


#Q6 STARTS HERE
data_q6 = data_set.iloc[:,91:182]
data_q6_energy = data_q6.mean(axis=1)
data_q6_ind = pd.concat([data_q6_energy, data_set.iloc[:, 215:217]], axis=1)
data_q6_independent = data_q6_ind.dropna()
data_q6_independent.columns = ['energy', 'age','gender']
removed_indices_q6 = data_q6.index.difference(data_q6_independent.index).values
data_art_q6 = data_set.iloc[:,0:91].drop(index=removed_indices_q6)
data_art_q6 = data_art_q6.mean(axis=1)


# In[1062]:


X_train_q6, X_test_q6, y_train_q6, y_test_q6 = train_test_split(data_q6_independent, data_art_q6, test_size=0.3,random_state=16790632)
model_q6 = LinearRegression()
model_q6.fit(X_train_q6, y_train_q6)
y_pred_q6 = model_q6.predict(X_test_q6)

# Calculate the RMSE
rmse_q6 = np.sqrt(mean_squared_error(y_test_q6, y_pred_q6))
r2_q6 = model_q6.score(data_q6_independent, data_art_q6)
print(f'RMSE: {rmse_q6}')
print("The rsquared is", r2_q6)


# In[1063]:


results_data_q6 = pd.DataFrame({'Actual': y_test_q6.to_numpy().flatten(), 'Predicted': y_pred_q6.flatten()})
results_data_q6.head(3)


# In[1064]:


graph_q6= results_data_q6.head(10)
graph_q6.plot(kind='bar')
plt.title('Predicting art prefence ratings from energy ratings and demographics', fontsize=15)
plt.ylabel('Mean rating for an art',fontsize=15)
plt.xlabel('Actual and predicted ratings',fontsize=10)


# In[1008]:


#Q7 STARTS HERE
data_art_q7 = data_set.iloc[:,0:91]
data_art_q7 = data_art_q7.mean(axis=0)
data_art_q7_numpy = data_art_q7.to_numpy()
data_energy_q7 = data_set.iloc[:,91:182]
data_energy_q7 = data_energy_q7.mean(axis=0)
data_energy_q7_numpy = data_energy_q7.to_numpy()
data_kmeans = np.column_stack((data_art_q7_numpy, data_energy_q7_numpy))
data_cluster = pd.DataFrame(data_kmeans, columns=["art", "energy"])
data_cluster = data_cluster.apply(zscore)
X_q7 = data_cluster[['art', 'energy']]


# In[1009]:


numClusters = 9 
sSum = np.empty([numClusters,1])*np.NaN # i
for ii in range(2, numClusters+2): 
    kMeans = KMeans(n_clusters = int(ii)).fit(X_q7) 
    cId = kMeans.labels_ 
    cCoords = kMeans.cluster_centers_ 
    s = silhouette_samples(X_q7,cId) # compute the mean silhouette coefficient of all samples
    sSum[ii-2] = sum(s) # take the sum
    # Plot data:
    plt.subplot(3,3,ii-1) 
    plt.hist(s,bins=20) 
    plt.xlim(-0.2,1)
    plt.ylim(0,250)
    plt.xlabel('Silhouette score')
    plt.ylabel('Count')
    plt.title('Sum: {}'.format(int(sSum[ii-2]))) # sum rounded to nearest integer
    plt.tight_layout() # adjusts subplot 


# In[1010]:


# Plot the sum of the silhouette scores as a function of the number of clusters, to make it clearer what is going on
plt.plot(np.linspace(2,numClusters,9),sSum)
plt.xlabel('Number of clusters')
plt.ylabel('Sum of silhouette scores')
plt.show()


# In[1011]:


data_kmeans_art = data_set.iloc[:,0:91].mean()
data_kmeans_energy = data_set.iloc[:,91:182].mean()
x = np.column_stack((data_kmeans_art, data_kmeans_energy))
numClusters = 4
kMeans = KMeans(n_clusters = numClusters).fit(x) 
cId = kMeans.labels_ 
cCoords = kMeans.cluster_centers_ 
# Plot the color-coded data:
for ii in range(numClusters):
    plotIndex = np.argwhere(cId == int(ii))
    plt.plot(x[plotIndex,0],x[plotIndex,1],'o',markersize=2)
    plt.plot(cCoords[int(ii-1),0],cCoords[int(ii-1),1],'o',markersize=5,color='black')  
    plt.xlabel('Art')
    plt.ylabel('Energy')
    plt.text(cCoords[ii,0], cCoords[ii,1], "Cluster {}".format(ii+1), fontsize=6)
    
plt.show()


# In[1012]:


data_cluster_1 = art.iloc[np.where(cId == 0)]


# In[1013]:


data_cluster_2 = art.iloc[np.where(cId == 1)]
data_cluster_2


# In[1014]:


data_cluster_3 = art.iloc[np.where(cId == 2)]
data_cluster_3


# In[1015]:


data_cluster_4 = art.iloc[np.where(cId == 3)]
data_cluster_4


# In[1016]:


#Q8
data_q8 = data_set.iloc[:, 205:215]
data_q8_nonan = data_q8.dropna()
data_eda_array = data_q8_nonan.to_numpy()
index_removal = data_q8.index.difference(data_q8_nonan.index).values
data_art_q8 = data_set.iloc[:,0:91]
data_art_q8 = data_art_q8.drop(index_removal)
data_art_q8 = data_art_q8.mean(axis=1)


# In[1017]:


r_q8 = np.corrcoef(data_eda_array,rowvar=False)
plt.imshow(r_q8) 
plt.colorbar()
plt.show()


# In[1018]:


zscoredData_q8 = stats.zscore(data_eda_array)

# Initialize PCA object and fit to our data:
pca_q8 = PCA().fit(zscoredData_q8)

# Eigenvalues: Single vector of eigenvalues in decreasing order of magnitude
eigVals_q8 = pca_q8.explained_variance_

# Loadings (eigenvectors): Weights per factor in terms of the original data.
loadings_q8 = pca_q8.components_

# Rotated Data - simply the transformed data:
origDataNewCoordinates_q8 = pca_q8.fit_transform(zscoredData_q8)


# In[ ]:





# In[ ]:





# In[1019]:


#Q8
rotatedData_q8 = pca_q8.fit_transform(zscoredData_q8)
varExplained_q8 = eigVals_q8/sum(eigVals_q8)*100
predictors_q8 = 10
x_q8 = np.linspace(1,predictors_q8,predictors_q8)
plt.bar(x_q8, eigVals_q8, color='gray')
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.show()


# In[1067]:


whichPrincipalComponent = 0 
plt.bar(x_q8,loadings_q8[whichPrincipalComponent,:]*-1)
plt.xlabel('Questions')
plt.ylabel('Loading')
plt.show()


# In[1069]:


whichPrincipalComponent = 1 
plt.bar(x_q8,loadings_q8[whichPrincipalComponent,:])
plt.xlabel('Questions')
plt.ylabel('Loading')
plt.show()
#q6 impornat to know for q10


# In[1070]:


#Using q1
data_q8_lm = data_q8_nonan.iloc[:, 0]
X_q8 =  data_q8_lm.to_numpy().reshape(286, -1)
y_q8 = data_art_q8.to_numpy().reshape(286,-1)
X_train_q8, X_test_q8, y_train_q8, y_test_q8 = train_test_split(X_q8, y_q8, test_size=0.3, random_state=16790632)
model_q8 = LinearRegression()
model_q8.fit(X_train_q8, y_train_q8)
y_pred_q8 = model_q8.predict(X_test_q8)
r2_q8 = r2_score(y_test_q8, y_pred_q8)
rmse_q8 = mean_squared_error(y_test_q8, y_pred_q8, squared=False)
print("RMSE:", rmse_q8)


# In[1071]:


#Q9
data_q9 = data_set.iloc[:,182:194]
data_q9_nonan = data_q9.dropna()
index_removal_q9 = data_q9.index.difference(data_q9_nonan.index).values
data_q9_array = data_q9_nonan.to_numpy()
data_art_q9 = data_set.iloc[:,0:91]
data_art_q9 = data_art_q9.drop(index_removal_q9)
data_art_q9 = data_art_q9.mean(axis=1)


# In[1072]:


r_q9 = np.corrcoef(data_q9_array,rowvar=False)
plt.imshow(r_q9) 
plt.colorbar()
plt.show()


# In[1073]:


zscoredData_q9 = stats.zscore(data_q9_array)

# Initialize PCA object and fit to our data:
pca_q9 = PCA().fit(zscoredData_q9)

# Eigenvalues: Single vector of eigenvalues in decreasing order of magnitude
eigVals_q9 = pca_q9.explained_variance_

# Loadings (eigenvectors): Weights per factor in terms of the original data.
loadings_q9 = pca_q9.components_

# Rotated Data - simply the transformed data:
origDataNewCoordinates_q9 = pca_q9.fit_transform(zscoredData_q9)


# In[1074]:


rotatedData_q9 = pca_q9.fit_transform(zscoredData_q9)
varExplained_q9 = eigVals_q9/sum(eigVals_q9)*100
predictors_q9 = 12
x_q9 = np.linspace(1,predictors_q9,predictors_q9)
plt.bar(x_q9, eigVals_q9, color='blue')

plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.show()


# In[1075]:


kaiserThreshold = 1
print('Number of factors selected by Kaiser criterion:', np.count_nonzero(eigVals_q9 > kaiserThreshold))


# In[1076]:


whichPrincipalComponent_q9 = 0 
plt.bar(x_q9,loadings_q9[whichPrincipalComponent_q9,:])
plt.xlabel('Questions')
plt.ylabel('Loading')
plt.show()


# In[1077]:


whichPrincipalComponent_q9 = 1 
plt.bar(x_q9,loadings_q9[whichPrincipalComponent_q9,:]*-1)
plt.xlabel('Questions')
plt.ylabel('Loading')
plt.show()


# In[1078]:


whichPrincipalComponent_q9 = 2 
plt.bar(x_q9,loadings_q9[whichPrincipalComponent_q9,:]*-1)
plt.xlabel('Questions')
plt.ylabel('Loading')
plt.show()


# In[1079]:


#after seeing the graph I can determine that we can use q1, q8, and q9 for lm model for this question.


# In[1080]:


data_q9_lm = data_q9_nonan.iloc[:, [0, 7, 8]]


# In[1081]:


X_train_q9, X_test_q9, y_train_q9, y_test_q9 = train_test_split(data_q9_lm, data_art_q9, test_size=0.3,random_state=16790632)

# Create a linear regression model and fit it to the training data
model_q9 = LinearRegression()
model_q9.fit(X_train_q9, y_train_q9)
y_pred_q9 = model_q9.predict(data_q9_lm)

# Calculate the RMSE
rmse_q9 = np.sqrt(mean_squared_error(data_art_q9, y_pred_q9))

print(f'RMSE: {rmse_q9}')


# In[1082]:


#Q10 STARTS HERE
data_political_q10 = data_set.iloc[:, 217:218]
data_political_q10.columns = ['Political']
data_political_q10


# In[1083]:


binary = pd.DataFrame(np.where(data_political_q10 <= 2, 0, 1), columns=['outcome'])

# concatenate the new dataframe with the original dataframe
binary


# In[1084]:


data_set_q10 = data_set.dropna()
dropped_indices_q10 = data_set.index.difference(data_set_q10.index)


# In[1085]:


data_art_q10 = data_set_q10.iloc[:,0:91].mean(axis=1)
data_energy_q10 = data_set_q10.iloc[:,91:182].mean(axis=1)


# In[1086]:


data_dark = data_set_q10.iloc[:,182:194]
#using pca from previous examples
data_dark_q10 = data_dark.iloc[:, [0, 7, 8]]
data_dark_q10


# In[1087]:


data_adventure = data_set_q10.iloc[:,194:205]
#we will have to do pca on this
data_q10_adventure = data_adventure.to_numpy()
r_action = np.corrcoef(data_q10_adventure,rowvar=False)
plt.imshow(r_action) 
plt.colorbar()
plt.show()


# In[1088]:


zscoredData_action = stats.zscore(data_q10_adventure)

# Initialize PCA object and fit to our data:
pca_action = PCA().fit(zscoredData_action)

# Eigenvalues: Single vector of eigenvalues in decreasing order of magnitude
eigVals_action = pca_action.explained_variance_

# Loadings (eigenvectors): Weights per factor in terms of the original data.
loadings_action = pca_action.components_

# Rotated Data - simply the transformed data:
origDataNewCoordinates_action = pca_action.fit_transform(zscoredData_action)


# In[1089]:


rotatedData_action = pca_action.fit_transform(zscoredData_action)
varExplained_action = eigVals_action/sum(eigVals_action)*100
predictors_action = 11
x_action = np.linspace(1,predictors_action,predictors_action)
plt.bar(x_action, eigVals_action, color='blue')

plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.show()


# In[1090]:


kaiserThreshold = 1
print('Number of factors selected by Kaiser criterion:', np.count_nonzero(eigVals_action > kaiserThreshold))


# In[1091]:


whichPrincipalComponent_action = 0 
plt.bar(x_action,loadings_action[whichPrincipalComponent_action,:]*-1)
plt.xlabel('Questions')
plt.ylabel('Loading')
plt.show()
#q6


# In[1092]:


whichPrincipalComponent_action = 1 
plt.bar(x_action,loadings_action[whichPrincipalComponent_action,:]*-1)
plt.xlabel('Questions')
plt.ylabel('Loading')
plt.show()
#q3


# In[1093]:


whichPrincipalComponent_action = 2 
plt.bar(x_action,loadings_action[whichPrincipalComponent_action,:]*-1)
plt.xlabel('Questions')
plt.ylabel('Loading')
plt.show()
#q5


# In[1094]:


data_q10_action = data_adventure.iloc[:, [2, 4, 5]]
data_q10_action


# In[1095]:


data_q10_image = data_set_q10.iloc[:,205:215]
#from earlier PCA we choose question 1 and 6 
data_q10_self_image = data_q10_image.iloc[:, [0, 5]]
data_q10_self_image


# In[1096]:


data_rest_q10 = data_set_q10.iloc[:,[215,216,218,219,220]]
data_rest_q10


# In[1097]:


data_combined_q10 = pd.concat([data_art_q10, data_energy_q10, data_dark_q10,data_q10_action,data_q10_self_image,data_rest_q10], axis=1, ignore_index=True)


# In[1098]:


data_combined_q10.head(5)


# In[1099]:


binary = binary.drop(dropped_indices_q10)


# In[1100]:


X_train_q10, X_test_q10, y_train_q10, y_test_q10 = train_test_split(data_combined_q10, binary, test_size=0.3, random_state=16790632)

# Train logistic regression model
logreg = LogisticRegression()
logreg.fit(X_train_q10, y_train_q10)

# Evaluate model on test set
accuracy = logreg.score(X_test_q10, y_test_q10)
print("Accuracy:", accuracy)
probs = logreg.predict_proba(X_test_q10)[:,1]

# Calculate ROC AUC score
auc = roc_auc_score(y_test_q10, probs)

print("ROC AUC:", auc)


# In[1101]:


y_pred_proba = logreg.predict_proba(X_test_q10)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test_q10,  y_pred_proba)

#create ROC curve
plt.plot(fpr,tpr,label='ROC curve')
plt.ylabel('True positive rate')
plt.plot([0, 1], [0, 1], 'k--', label='No Predictive Power')
plt.xlabel('False Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# In[ ]:





# In[ ]:




