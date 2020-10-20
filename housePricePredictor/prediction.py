#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Compulsory Attributes: Price, Area, No. of Bedrooms, Resale, Gymnasium, SwimmingPool,
#LandscapedGardens, JoggingTrack, Intercom, LiftAvailable, Gasconnection, CarParking,
#24X7Security, School

#Sports: IndoorGames + SportsFacility + ClubHouse + Children'splayarea + GolfCourse

#Merge attributes: MaintenanceStaff + RainWaterHarvesting + ShoppingMall + Hospital + Cafeteria + StaffQuarter + PowerBackup + School + ATM

#Facilities: WashingMachine + AC + Wifi + BED + Microwave + TV + DiningTable + Sofa + Wardrobe + Refrigerator

#Removing: Multipurpose, Vastu and Location(Temp)


# In[ ]:





# In[1]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


# In[75]:


def read_data(i, city):
    file = city + ".csv"
    df = pd.read_csv(file)
    df = clean(df)

    j = 0
    df1 = pd.DataFrame()
    df1["Price"] = pd.Series(df["Price"])
    while j <= i:
        if j == 0:
            df1["Area"] = pd.Series(df["Area"])
            df1["No. of Bedrooms"] = pd.Series(df["No. of Bedrooms"])
            df1["Resale"] = pd.Series(df["Resale"])
        elif j == 1:
            df["Merge1"] = df["Intercom"] + df["24X7Security"] + df["PowerBackup"] + df["CarParking"] + df["LiftAvailable"]
            df1["Merge1"] = pd.Series(df["Merge1"])
            df.drop(['Intercom', '24X7Security', 'CarParking', 'LiftAvailable', 'PowerBackup'], axis=1, inplace=True)
        elif j == 2:
            df["Merge2"] = df["MaintenanceStaff"] + df["RainWaterHarvesting"] + df["Gymnasium"] + df["SwimmingPool"] + df["LandscapedGardens"] + df["JoggingTrack"]
            df1["Merge2"] = pd.Series(df["Merge2"])
        elif j == 3:
            df["Sports"] = df["IndoorGames"] + df["SportsFacility"] + df["ClubHouse"] + df["Children'splayarea"] + df["GolfCourse"]
            df1["Sports"] = pd.Series(df["Sports"])
            df.drop(['IndoorGames', 'SportsFacility', 'ClubHouse', "Children'splayarea", 'GolfCourse'], axis=1, inplace=True)
        elif j == 4:
            df["Facilities1"] = df["AC"] + df["Wifi"] + df["TV"] + df["Microwave"]
            df1["Facilities1"] = pd.Series(df["Facilities1"])
            df.drop(['AC', 'Wifi', 'TV', 'Microwave'], axis=1, inplace=True)
        elif j == 5:
            df["Facilities2"] = df["BED"]  + df["DiningTable"] + df["Wardrobe"] + df["Refrigerator"]
            df1["Facilities2"] = pd.Series(df["Facilities2"])
            df.drop(['WashingMachine', 'BED', 'DiningTable', 'Sofa', 'Wardrobe', 'Refrigerator'], axis=1, inplace=True)
        j += 1

    #df.drop(['MaintenanceStaff', 'RainWaterHarvesting', 'ShoppingMall', 'Hospital', 'Cafeteria', 'StaffQuarter', 'PowerBackup', 'School', 'ATM'], axis=1, inplace=True)

    df.drop(['MultipurposeRoom', 'VaastuCompliant', 'Location'], axis=1, inplace=True)
    print(df1.head())
    normalize(df1)


# In[76]:
def clean(df):
    return df.replace(9,np.NaN).dropna()

# min max normalization 
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


# In[82]:

def splitt(df):
    y = np.array(df['Price']).reshape(-1, 1)
    df.drop('Price', axis = 1, inplace = True)
    X = np.array(df)
    ones = np.ones([X.shape[0], 1])
    X = np.concatenate([ones, X],1)

    #print(X.shape)
    global X_train, X_test, y_train, y_test, alpha, iters, gamma, theta, mtrain, mtest, c
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 3/10)

    alpha = 0.0001
    iters = 1000
    gamma = 9000
    #theta = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    theta = np.zeros([1, X.shape[1]])
    #print(theta.shape)
    mtrain = X_train.shape[0]
    mtest = X_test.shape[0]
    c = []
    #print(y)

    a = gradient_descent()
    print(a)



# In[83]:


def cost():
   global X_train, X_test, y_train, y_test, alpha, iters, gamma, theta, mtrain, mtest, cost
   hx = np.dot(X_train, theta.T)
   g = np.sum(np.dot(theta.T, theta))
   Jtheta = (1/2*mtrain) * (np.sum((hx - y_train)**2) + (gamma/2*mtrain)*g)
   c.append(Jtheta)
   #print(Jtheta)


# In[84]:


def gradient_descent():
    global X_train, X_test, y_train, y_test, alpha, iters, gamma, theta, mtrain, mtest, cost
    for i in range(iters):
        cost()
        hx = np.dot(X_train, theta.T)
        dJtheta = np.dot((hx - y_train).T, X_train)
        theta = theta*(1 - ((gamma*alpha)/mtrain)) - ((alpha/mtrain) * dJtheta)
        #print(theta)
    return theta


# In[ ]:





# In[ ]:





# In[ ]:





# In[110]:


def caller(i, city, Area, Bedroom, Resale, Intercom = None, Security = None, PowerBackup = None, CarParking = None, Lift = None, MaintenanceStaff = None, Gymnasium = None, SwimmingPool = None, Landscape = None, Jogging = None, RainWater = None, SportsFacility = None, Clubhouse = None, IndoorGames = None, Childrenplay = None, BED = None, Diningtable = None, Wardrobe = None, Referigator = None, AC = None, Wifi = None, TV = None, Microwave = None):
    read_data(i, city)

    a = theta
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

    if i == 0:
        pri = a[0, 0] + a[0, 1]*float(Area) + a[0, 2]*float(Bedroom) + a[0, 3]*float(Resale)
    elif i == 1:
        pri = a[0, 0] + a[0, 1]*float(Area) + a[0, 2]*float(Bedroom) + a[0, 3]*float(Resale) + a[0, 4]*c1
    elif i == 2:
        pri = a[0, 0] + a[0, 1]*float(Area) + a[0, 2]*float(Bedroom) + a[0, 3]*float(Resale) + a[0, 4]*c1 + a[0, 5]*c2
    elif i == 3:
        pri = a[0, 0] + a[0, 1]*float(Area) + a[0, 2]*float(Bedroom) + a[0, 3]*float(Resale) + a[0, 4]*c1 + a[0, 5]*c2 + a[0, 6]*c3
    elif i == 4:
        pri = a[0, 0] + a[0, 1]*float(Area) + a[0, 2]*float(Bedroom) + a[0, 3]*float(Resale) + a[0, 4]*c1 + a[0, 5]*c2 + a[0, 6]*c3 + a[0, 7]*c4
    else:
        pri = a[0, 0] + a[0, 1]*float(Area) + a[0, 2]*float(Bedroom) + a[0, 3]*float(Resale) + a[0, 4]*c1 + a[0, 5]*c2 + a[0, 6]*c3 + a[0, 7]*c4 + a[0, 8]*c5

    #a = theta
    #pri = a[0, 0] + a[0, 1]*720 + a[0, 2]*1 + a[0, 3]*1 + a[0, 4]*0 + a[0, 5]*0 + a[0, 6]*0 + a[0, 7]*0 + a[0, 8]*0 + a[0, 9]*1 + a[0, 10]*1 + a[0, 11]*0 + a[0, 12]*1 + a[0, 13]*0 + a[0, 14]*2 + a[0, 15]*0
    global minn, maxx
    pri = (pri * (maxx - minn)) + minn
    print(pri/10)
    return int(pri/12) 


# In[1]:


# caller(1, "Mumbai", 700, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
# i, city, Area, Bedroom, Resale, Intercom = None, Security = None, PowerBackup = None, CarParking = None, Lift = None,
# MaintenanceStaff = None, Gymnasium = None, SwimmingPool = None, Landscape = None, Jogging = None, RainWater = None,
# SportsFacility = None, Clubhouse = None, IndoorGames = None, Childrenplay = None, BED = None, Diningtable = None,
# Wardrobe = None, Referigator = None, AC = None, Wifi = None, TV = None, Microwave = None
# In[ ]:





# In[ ]:
