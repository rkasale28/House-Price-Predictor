#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
import os

# In[16]:


def read_data(i, city):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(BASE_DIR, 'predictor/data',city+'.csv'))
    
    global mapping_index
    df["Location"], mapping_index = pd.Series(df["Location"]).factorize()      
    j = 0
    
    df = clean(df)
    df1 = pd.DataFrame()
    df1["Price"] = pd.Series(df["Price"])
    
    while j <= i:
        if j is 0:
            df1["Area"] = pd.Series(df["Area"])
            df1["No. of Bedrooms"] = pd.Series(df["No. of Bedrooms"])
            df1["Resale"] = pd.Series(df["Resale"])
            df1["Location"] = pd.Series(df["Location"])
        elif j is 1:
            df["Merge1"] = df["Intercom"] + df["24X7Security"] + df["PowerBackup"] + df["CarParking"] + df["LiftAvailable"]
            df1["Merge1"] = pd.Series(df["Merge1"])
            df.drop(['Intercom', '24X7Security', 'CarParking', 'LiftAvailable', 'PowerBackup'], axis=1, inplace=True)
        elif j is 2:
            df["Merge2"] = df["MaintenanceStaff"] + df["RainWaterHarvesting"] + df["Gymnasium"] + df["SwimmingPool"] + df["LandscapedGardens"] + df["JoggingTrack"] 
            df1["Merge2"] = pd.Series(df["Merge2"])
        elif j is 3:
            df["Sports"] = df["IndoorGames"] + df["SportsFacility"] + df["ClubHouse"] + df["Children'splayarea"] + df["GolfCourse"]
            df1["Sports"] = pd.Series(df["Sports"])
            df.drop(['IndoorGames', 'SportsFacility', 'ClubHouse', "Children'splayarea", 'GolfCourse'], axis=1, inplace=True)
        elif j is 4: 
            df["Facilities1"] = df["AC"] + df["Wifi"] + df["TV"] + df["Microwave"]
            df1["Facilities1"] = pd.Series(df["Facilities1"])
            df.drop(['AC', 'Wifi', 'TV', 'Microwave'], axis=1, inplace=True)
        elif j is 5:
            df["Facilities2"] = df["BED"]  + df["DiningTable"] + df["Wardrobe"] + df["Refrigerator"]
            df1["Facilities2"] = pd.Series(df["Facilities2"])
            df.drop(['WashingMachine', 'BED', 'DiningTable', 'Sofa', 'Wardrobe', 'Refrigerator'], axis=1, inplace=True)
        j += 1
    
    df.drop(['MultipurposeRoom', 'VaastuCompliant'], axis=1, inplace=True)
    remove_outliers(df1)


# In[17]:


def clean(df):
    return df.replace(9,np.NaN).dropna()


# In[18]:


def remove_outliers(df):
    df = df.sort_values(['Price'])
    len  = int(df.shape[0]/2)
    df1 = df[:len]
    df2 = df[len:]

    q1 = df1.median()['Price']
    q2 = df.median()['Price']
    q3 = df2.median()['Price']
    iqr = q3 - q1

    upper_limit = q1- 1.5*iqr
    lower_limit = q3 + 1.5*iqr

    df = df[(df['Price']>upper_limit) & (df['Price']<lower_limit)]
    normalize(df)


# In[19]:


def normalize(df):  
    result = df.copy()
    global maxx
    global minn
    i = 0
    for feature_name in df.columns:
        max_val = df[feature_name].max()
        min_val = df[feature_name].min()
        result[feature_name] = ((df[feature_name] - min_val) / (max_val - min_val))
        if i == 0:
            maxx = max_val
            minn = min_val
            i = 1
    
    df = result.copy()
    #print(maxx, minn)
    #print(df.head())
    splitt(df)


# In[20]:


def splitt(df):
    y = np.array(df['Price']).reshape(-1, 1)
    df.drop('Price', axis = 1, inplace = True)
    X = np.array(df)
    ones = np.ones([X.shape[0], 1]) 
    X = np.concatenate([ones, X],1) 
    
    #print(X.shape)
   
    global X_train, X_test, y_train, y_test, alpha, iters, gamma, theta, mtrain, mtest, c
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 3/10)
    
    alpha = 0.06 #learning rate
    iters = 1000
    theta = np.zeros([1, X.shape[1]]) 
    #print(theta.shape)
    mtrain = X_train.shape[0]
    mtest = X_test.shape[0]
    c = []
    #print(y)
    
    a = gradient_descent()
    print(a)
    


# In[21]:


def cost():
   global X_train, X_test, y_train, y_test, alpha, iters, gamma, theta, mtrain, mtest, cost
   tobesummed = np.power(((X_train @ theta.T)-y_train),2)
   summm = np.sum(tobesummed)/(2 * mtrain)
   c.append(summm)


# In[22]:


def gradient_descent():
    global X_train, X_test, y_train, y_test, alpha, iters, gamma, theta, mtrain, mtest, cost
    for i in range(iters):
        cost()
        theta = theta - (alpha/mtrain) * np.sum(X_train * (X_train @ theta.T - y_train), axis=0)
    return theta


# In[23]:


def rms():
    global X_test, theta, y_test, mtest
    summ = 0
    hx = np.dot(X_test, theta.T) 
    diff = y_test - hx
    rmse = (np.sqrt(np.sum(np.square(diff))))/mtest #Root mean Square error
    mae = (np.sum(np.abs(diff)))/mtest #Mean absolute error
    print("Root Mean Square Error: ",rmse)
    print("Mean Absolute Error: ", mae)


# In[38]:


def caller(i, city, Area, Bedroom, Resale, Location, Intercom = None, Security = None, PowerBackup = None, CarParking = None, Lift = None, MaintenanceStaff = None, Gymnasium = None, SwimmingPool = None, Landscape = None, Jogging = None, RainWater = None, SportsFacility = None, Clubhouse = None, IndoorGames = None, Childrenplay = None, BED = None, Diningtable = None, Wardrobe = None, Referigator = None, AC = None, Wifi = None, TV = None, Microwave = None):
    read_data(i, city)
    
    a = theta
    a = a/100
    a = a/15
    j = 0
    c1 = 0
    c2 = 0
    c3 = 0 
    c4 = 0
    c5 = 0
    
    while j <= i:
        if j == 1:
            if Intercom:
                c1 += 1
            if Security:
                c1 += 1
            if PowerBackup:
                c1 += 1
            if CarParking:
                c1 += 1
            if Lift:
                c1 += 1 
        
        if j == 2:
            if MaintenanceStaff:
                c2 += 1
            if Gymnasium:
                c2 += 1
            if SwimmingPool:
                c2 += 1
            if Landscape:
                c2 += 1
            if Jogging:
                c2 += 1
            if RainWater:
                c2 += 1
                
        if j == 3:
            if SportsFacility:
                c3 += 1
            if Clubhouse:
                c3 += 1
            if IndoorGames:
                c3 += 1
            if Childrenplay:
                c3 += 1
                
        if j == 4:
            if BED:
                c4 += 1
            if Diningtable:
                c4 += 1
            if Wardrobe:
                c4 += 1
            if Referigator:
                c4 += 1
                
        if j == 5:
            if AC:
                c5 += 1
            if Wifi:
                c5 += 1
            if TV:
                c5 += 1
            if Microwave:
                c5 += 1
            
        j += 1
        
    global mapping_index
    x = mapping_index.get_loc(Location)
        
    
    if i == 0:
        pri = a[0, 0] + a[0, 1]*float(Area) + a[0, 2]*float(Bedroom) + a[0, 3]*float(Resale) + a[0, 4]*x
    elif i == 1:
        pri = a[0, 0] + a[0, 1]*float(Area) + a[0, 2]*float(Bedroom) + a[0, 3]*float(Resale) + a[0, 4]*x + a[0, 5]*c1
    elif i == 2:
        pri = a[0, 0] + a[0, 1]*float(Area) + a[0, 2]*float(Bedroom) + a[0, 3]*float(Resale) + a[0, 4]*x + a[0, 5]*c1 + a[0, 6]*c2
    elif i == 3:
        pri = a[0, 0] + a[0, 1]*float(Area) + a[0, 2]*float(Bedroom) + a[0, 3]*float(Resale) + a[0, 4]*x + a[0, 5]*c1 + a[0, 6]*c2 + a[0, 7]*c3
    elif i == 4:
        pri = a[0, 0] + a[0, 1]*float(Area) + a[0, 2]*float(Bedroom) + a[0, 3]*float(Resale) + a[0, 4]*x + a[0, 5]*c1 + a[0, 6]*c2 + a[0, 7]*c3 + a[0, 8]*c4
    else:
        pri = a[0, 0] + a[0, 1]*float(Area) + a[0, 2]*float(Bedroom) + a[0, 3]*float(Resale) + a[0, 4]*x + a[0, 5]*c1 + a[0, 6]*c2 + a[0, 7]*c3 + a[0, 8]*c4 + a[0, 9]*c5
    
    global minn, maxx    
    pri = (pri * (maxx - minn)) + minn
    print(pri)
    rms()
    return pri
    


# In[37]:


# caller(1, "Mumbai", 720, 1, 1, "Kharghar")


# In[31]:


def output():
    global c, iters
    fig, ax = plt.subplots()  
    ax.plot(np.arange(iters), c, 'r')  
    ax.set_xlabel('Iterations')  
    ax.set_ylabel('Cost')  
    ax.set_title('Error vs. Training Epoch')


# In[33]:


# rms() 


# In[ ]:





# In[ ]:




