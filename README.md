# Electric-Vehicle-Internship code

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
url=r"C:\Users\home\Downloads\Indian_automoble_buying_behavour_study_1.0.csv"
df=pd.read_csv(url)
print(df.head())
EV_df=df.drop(['Marrital Status','Education','Wife Working'],axis=1)
print(EV_df)
print(df.shape)
print(df.isnull().sum())
print(df)
print(EV_df)
EV_df["Profession"]= EV_df["Profession"].map({"Salaried" : 0, "Business" : 1})
print(EV_df.head())
EV_df["Personal loan"]= EV_df["Personal loan"].map({"No" : 0, "Yes" : 1})
print(EV_df.head())
EV_df["House Loan"]= EV_df["House Loan"].map({"No" : 0, "Yes" : 1})
print(EV_df.head())
EV_df["Make"]= EV_df["Make"].map({"Ciaz" : 0, "SUV" : 1, "Baleno": 2, "Creta": 3, "City": 4, "i20": 5,"Duster": 6, "Verna": 7, "Luxuray": 8})
print(EV_df.head())
print(plt.figure(1, figsize=(15,6)))
n=0
for x in ["Age","Total Salary","Price"]:
    n +=1
    print(plt.subplot(1,3,n))
    print(plt.subplots_adjust(hspace=0.5,wspace=0.5))
    print(sns.distplot(EV_df[x], bins=20))
    print(plt.title("distplot of {} ".format(x)))
print(plt.show())
print(plt.figure(figsize=(15,5)))
sns.countplot(y="Profession",data=EV_df)
print(plt.show())
print(plt.figure(figsize=(15,5)))
sns.countplot(y="Make",data=EV_df)
print(plt.show())
print(plt.figure(figsize=(15,7)))
n=0
for cols in ["Age","Total Salary","Price"]:
    n+=1
    print(sns.set(style="whitegrid"))
    print(plt.subplot(1,3,n))
    print(plt.subplots_adjust(hspace=0.5,wspace=0.5))
    print(sns.violinplot(x=cols, y="Profession", data=EV_df))
    print(plt.ylabel("Profession" if n==1 else ''))
    print(plt.title("Violin Plot"))
print(plt.show())
age_18_25= EV_df.Age[(EV_df.Age >= 18) & (EV_df.Age<=25)]
age_26_35= EV_df.Age[(EV_df.Age >= 26) & (EV_df.Age<=35)]
age_36_45= EV_df.Age[(EV_df.Age >= 36) & (EV_df.Age<=45)]
age_45above= EV_df.Age[(EV_df.Age >= 46)]

agex=["18-25","26-35","36-45","45+"]
agey=[len(age_18_25.values),len(age_26_35.values),len(age_36_45.values),len(age_45above.values)]

print(plt.figure(figsize=(15,6)))
print(sns.barplot(x=agex,y=agey, palette="mako"))
print(plt.title("Number of Customers and Ages"))
print(plt.xlabel("Age"))
print(plt.ylabel("Number of Customers"))
print(plt.show())
print(sns.relplot(x="Total Salary",y="Price",data=EV_df))
X1=EV_df.loc[:,["Age","Price"]].values
from sklearn.cluster import KMeans
wcss=[]
for k in range(1,11):
    kmeans=KMeans(n_clusters=k, init="k-means++")
    kmeans.fit(X1)
    wcss.append(kmeans.inertia_)
print(plt.figure(figsize=(12,6)))
print(plt.grid())
print(plt.plot(range(1,11),wcss,linewidth=2, color='red',marker='8'))
print(plt.xlabel("K Value"))
print(plt.ylabel("WCSS"))
print(plt.show())
kmeans=KMeans(n_clusters=4)
label=kmeans.fit_predict(X1)
print(label)
print(kmeans.cluster_centers_)
plt.scatter(X1[:,0], X1[:,1], c=kmeans.labels_, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black')
print(plt.title("Clusters of Customers"))
print(plt.xlabel("Age"))
print(plt.ylabel("Price"))
print(plt.show())

X2=EV_df.loc[:,["Total Salary","Price"]].values
from sklearn.cluster import KMeans
wcss=[]
for k in range(1,11):
    kmeans=KMeans(n_clusters=k, init="k-means++")
    kmeans.fit(X2)
    wcss.append(kmeans.inertia_)
print(plt.figure(figsize=(12,6)))
print(plt.grid())
print(plt.plot(range(1,11),wcss,linewidth=2, color='red',marker='8'))
print(plt.xlabel("K Value"))
print(plt.ylabel("WCSS"))
print(plt.show())
kmeans=KMeans(n_clusters=5)
label=kmeans.fit_predict(X2)
print(label)
print(kmeans.cluster_centers_)
plt.scatter(X2[:,0], X2[:,1], c=kmeans.labels_, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black')
print(plt.title("Clusters of Customers"))
print(plt.xlabel("Total Salary"))
print(plt.ylabel("Price"))
print(plt.show())
