import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import os

def read_data(i, city):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(BASE_DIR, 'predictor/data',city+'.csv'))
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
        elif j == 2:
            df["Merge2"] = df["MaintenanceStaff"] + df["RainWaterHarvesting"] + df["Gymnasium"] + df["SwimmingPool"] + df["LandscapedGardens"] + df["JoggingTrack"]
            df1["Merge2"] = pd.Series(df["Merge2"])
        elif j == 3:
            df["Sports"] = df["IndoorGames"] + df["SportsFacility"] + df["ClubHouse"] + df["Children'splayarea"] + df["GolfCourse"]
            df1["Sports"] = pd.Series(df["Sports"])
        elif j == 4:
            df["Facilities1"] = df["AC"] + df["Wifi"] + df["TV"] + df["Microwave"]
            df1["Facilities1"] = pd.Series(df["Facilities1"])
        elif j == 5:
            df["Facilities2"] = df["BED"]  + df["DiningTable"] + df["Wardrobe"] + df["Refrigerator"]
            df1["Facilities2"] = pd.Series(df["Facilities2"])
        j += 1

    print(df1.head())
    remove_outliers(df1)

def clean(df):
    return df.replace(9,np.NaN).dropna()

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
    splitt(df)

# In[82]:

def splitt(df):
    y = np.array(df['Price']).reshape(-1, 1)
    df.drop('Price', axis = 1, inplace = True)
    X = np.array(df)
    ones = np.ones([X.shape[0], 1])
    X = np.concatenate([ones, X],1)

    global X_train, X_test, y_train, y_test, alpha, iters, gamma, theta, mtrain, mtest, c
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 3/10)

    alpha = 0.0001
    iters = 1000
    gamma = 9000
    theta = np.zeros([1, X.shape[1]])
    mtrain = X_train.shape[0]
    mtest = X_test.shape[0]
    c = []

    a = gradient_descent()
    print(a)

def cost():
   global X_train, X_test, y_train, y_test, alpha, iters, gamma, theta, mtrain, mtest, cost
   hx = np.dot(X_train, theta.T)
   g = np.sum(np.dot(theta.T, theta))
   Jtheta = (1/2*mtrain) * (np.sum((hx - y_train)**2) + (gamma/2*mtrain)*g)
   c.append(Jtheta)

def gradient_descent():
    global X_train, X_test, y_train, y_test, alpha, iters, gamma, theta, mtrain, mtest, cost
    for i in range(iters):
        cost()
        hx = np.dot(X_train, theta.T)
        dJtheta = np.dot((hx - y_train).T, X_train)
        theta = theta*(1 - ((gamma*alpha)/mtrain)) - ((alpha/mtrain) * dJtheta)
    return theta

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

    global minn, maxx
    pri = (pri * (maxx - minn)) + minn
    print(pri/12)
    return int(pri/12)


# caller(1, "Mumbai", 700, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
# i, city, Area, Bedroom, Resale, Intercom = None, Security = None, PowerBackup = None, CarParking = None, Lift = None,
# MaintenanceStaff = None, Gymnasium = None, SwimmingPool = None, Landscape = None, Jogging = None, RainWater = None,
# SportsFacility = None, Clubhouse = None, IndoorGames = None, Childrenplay = None, BED = None, Diningtable = None,
# Wardrobe = None, Referigator = None, AC = None, Wifi = None, TV = None, Microwave = None
