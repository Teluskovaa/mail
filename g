#7th
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import sklearn.metrics as sm
from sklearn.mixture import GaussianMixture
import pandas as pd
import numpy as np
iris = datasets.load_iris()
temp = pd.DataFrame(iris.data)
X=temp.sample(frac=1)
wcss=[]
for i in range(1,10):
 km=KMeans(n_clusters=i)
 km.fit(X)
 wcss.append(km.inertia_)
plt.figure(figsize=(12,6))
plt.plot(range(1,10),wcss, linewidth=2, color="red", marker ="8")
plt.xlabel("K Value")
plt.xticks(np.arange(1,10,1))
plt.ylabel("WCSS")
plt.show()
X.columns = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']
y = pd.DataFrame(iris.target)
y.columns = ['Targets']
model = KMeans(n_clusters=3)
model.fit(X)
score1=sm.accuracy_score(y, model.labels_)
print("Accuracy of KMeans=",score1)
plt.figure(figsize=(7,7))
colormap = np.array(['red', 'lime', 'black'])
plt.subplot(1, 2, 1)
plt.scatter(X.Sepal_Length, X.Sepal_Width, c=colormap[model.labels_], s=40)
plt.title('K Mean Classification')
gmm = GaussianMixture(n_components=3)
gmm.fit(X)
y_cluster_gmm = gmm.predict(X)
score2=sm.accuracy_score(y, y_cluster_gmm)
print("Accuracy of EM=",score2)
plt.subplot(1, 2, 2)
plt.scatter(X.Sepal_Length, X.Sepal_Width, c=colormap[y_cluster_gmm], s=40)
plt.title('EM Classification') 
