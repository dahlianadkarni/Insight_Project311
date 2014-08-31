#!/Users/dahlia/Documents/insight_virtual_env/bin/python
import sys
import pandas as pd
import numpy as np
import csv
import pymysql as mdb
import pandas.io.sql as psql
import matplotlib.pyplot as plt
import matplotlib
from sklearn import preprocessing, linear_model
from sklearn import gaussian_process
from datetime import datetime 

# Function to extract features (both categorical and linear)
def extract_features(Xtimes,resolution):
    if resolution =='D':
        f_year = Xtimes.year
        f_quarter = Xtimes.quarter
        f_month = Xtimes.month

        f_weekno = Xtimes.weekofyear
        f_dayofyr = Xtimes.dayofyear
        f_weekday = Xtimes.weekday
        f_day = Xtimes.day

        # Get categorical features
        g_year = pd.get_dummies(f_year)
        g_quarter= pd.get_dummies(f_quarter)
        g_month = pd.get_dummies(f_month)
        g_weekno = pd.get_dummies(f_weekno)
        g_dayofyr = pd.get_dummies(f_dayofyr)
        g_weekday = pd.get_dummies(f_weekday)
        g_day = pd.get_dummies(f_day)

        # Check if weekend or weekday
        weekend = pd.DataFrame((f_weekday>0) & (f_weekday<6).astype(int))

        # Convert linear features to dataframe
        f_year = pd.DataFrame(f_year)
        f_quarter = pd.DataFrame(f_quarter)
        f_month = pd.DataFrame(f_month)
        f_weekno = pd.DataFrame(f_weekno)
        f_dayofyr = pd.DataFrame(f_dayofyr)
        f_weekday = pd.DataFrame(f_weekday)
        f_day = pd.DataFrame(f_day)   

        # feature_full = pd.concat([g_year,g_quarter,g_month,g_weekno,g_dayofyr,g_weekday,g_day], axis=1)  
        # # The day of the year feature gives weird results, weekno not high a correlation
        feature_full = pd.concat([g_year,g_quarter,g_month,g_weekno,g_weekday,g_day], axis=1)  
        # feature_full = pd.concat([g_year,g_quarter,g_month,g_weekday,g_day], axis=1) 

        # # Adding linear features and weekday/weekend category
        #  feature_full = pd.concat([feature_full, weekend, f_year, f_quarter,f_month,f_weekno,f_dayofyr,f_weekday, f_day], axis=1) 
        # # Last 4 features not useful
        feature_full = pd.concat([feature_full, weekend, f_year, f_quarter,f_month], axis=1) 

#         feature_full = pd.concat([g_year,g_quarter,g_month,g_weekno,g_dayofyr,g_weekday,g_day], axis=1)  
#         feature_full = pd.concat([feature_full, weekend, f_year, f_quarter,f_month,f_weekno,f_dayofyr,f_weekday, f_day], axis=1) 
    else:
        f_year = Xtimes.year
        f_quarter = Xtimes.quarter
        f_month = Xtimes.month
        f_weekno = Xtimes.weekofyear


        # Get categorical features
        g_year = pd.get_dummies(f_year)
        g_quarter= pd.get_dummies(f_quarter)
        g_month = pd.get_dummies(f_month)
        g_weekno = pd.get_dummies(f_weekno)

        # Convert linear features to dataframe
        f_year = pd.DataFrame(f_year)
        f_quarter = pd.DataFrame(f_quarter)
        f_month = pd.DataFrame(f_month)
        f_weekno = pd.DataFrame(f_weekno)
    
        if resolution == 'W':
            feature_full = pd.concat([g_year,g_quarter,g_month,g_weekno], axis=1)  
            # feature_full = pd.concat([g_year,g_quarter,g_month], axis=1)  
            feature_full = pd.concat([feature_full, f_year, f_quarter,f_month,f_weekno], axis=1) 
        elif resolution == 'M':
            feature_full = pd.concat([g_year,g_quarter,g_month], axis=1)  
            feature_full = pd.concat([feature_full, f_year, f_quarter,f_month], axis=1) 
    feature_full.applymap(np.int) 
    return feature_full

# Linear regression fit and returns Z-score    
def my_linfit(X_train,Y_train, X_test, Y_test,timeindex,linregmodel1):
    linregmodel1.fit(X_train,Y_train)
    # cf = linregmodel.coef_
    Y_hat_train = linregmodel1.predict(X_train)
    Y_hat_test = linregmodel1.predict(X_test)

    n=len(Y_train)
    sigma_hat =float(((((Y_hat_train-Y_train)**2).sum())/(n-2))**0.5)
    sigma = sigma_hat*(((X_test.values-X_train.mean().values)**2).sum(axis=1)/((X_train.values-X_train.mean().values)**2).sum(axis=1).sum(axis=0)+1/n+1)**0.5
    
    return Y_hat_test, sigma

# Gaussian processes fit and returns Z-score  
def my_gpfit(X_train,Y_train, X_test, Y_test,timeindex,gp):
    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(X_train,Y_train) 

    # Make the prediction (ask for MSE as well)
    Y_hat_test, MSE = gp.predict(X_test, eval_MSE=True)
    sigma = np.sqrt(MSE)

    return Y_hat_test, sigma


# Find out the top complaints (in terms of volume) for a given borough and period
def topcomplaint(borough,period):
    if period == 1:
        period_str = '2014-07-01'
        dbtable = 'complaints_201407'
    else:
        period_str = '2014-05-01'
        dbtable = 'complaints_recent'
    db_test = mdb.connect(host='localhost',user='root',passwd='dahlia',db='db311_recent_2014')
    with db_test:
        query = "SELECT complaint_type,count(*) AS counts FROM " + dbtable + " WHERE created_date IS NOT NULL AND borough = '" + borough + "' GROUP BY complaint_type ORDER BY counts DESC  LIMIT 10;"
        topcomplaints_df = psql.read_sql(query, con=db_test)
    return topcomplaints_df
    
# function that takes a borough, period, and time-resolution, and outputs a list of top complaints by priority(z-score)    
def get_zscore(borough,period,res):
    if period ==1:
        table_test = 'complaints_201407'
        period_str = '2014-07-01'
        table_train = 'complaints1'
    elif period==3:    
        table_test = 'complaints_recent'
        period_str = '2014-05-01'
        table_train = 'complaints3'
    
    topcomplaints_df = topcomplaint(borough,period)    
    topcomplaints = topcomplaints_df['complaint_type']
    listtopcomplaints = list(topcomplaints_df['complaint_type'])
        
    mystring = "('" + "', '".join(listtopcomplaints) + "')"
        
    TopN = len(listtopcomplaints) 
    Z = np.zeros(TopN)
    for i in range(TopN):
        complaint_here = topcomplaints[i].replace('-','_').replace(' ','_').replace('\n','_').replace('*','').replace('(','').replace(')','').replace('/','_')
        filenametest = "311PKLData/"+borough + "_test" +str(period)+"_"+ complaint_here + ".pkl"
        cv_test = pd.read_pickle(filenametest)
        filenametrain = "311PKLData/"+borough + "_train"+str(period)+"_" + complaint_here + ".pkl"
        cv_train = pd.read_pickle(filenametrain)  
        
        cv_test = cv_test.sort_index().resample(res,how=sum)
        cv_train = cv_train.sort_index().resample(res,how=sum)
              
        L = len(cv_test)
        cv = pd.concat([cv_train,cv_test])
        X = extract_features(cv.index,res)
        X_train = X[:-L]
        X_test = X[-L:]
        Y_train = cv_train.values
        Y_test = cv_test.values
        timeindex = cv_test.index
        
        if len(cv_train[(cv_train.index > '2011-05-01') & (cv_train.index < '2013-05-01')])<5:
            Z[i]=float('NaN')
            continue
        
        # # If you want to infer alpha for Ridge regression via cross validation
        linregmodel0 = linear_model.RidgeCV (alphas=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0])
        linregmodel0.fit(X_train,Y_train)
        alp = linregmodel0.alpha_ 
        # alp = 0.5
# 
        # # Linear regression - OLS or Ridge or Lasso
#         linregmodel = linear_model.LinearRegression()
        linregmodel = linear_model.Ridge (alpha = alp)
#         linregmodel = linear_model.Lasso(alpha = 10)
# 
#         Y_hat_test, sigma = my_linfit(X_train,Y_train, X_test, Y_test, cv_test.index,linregmodel)
#          
        # # Gaussian Processes
        gp = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
        Y_hat_test, sigma = my_gpfit(X_train,Y_train, X_test, Y_test,timeindex,gp)
        # # # Optional noise feature
#         gp = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
#         y=Y_train
#         dy = 0.5 + 1.0 * np.random.random(y.shape)
#         noise = np.random.normal(0, dy)
#         y += noise
#         Y_hat_test, sigma = my_gpfit(X_train,y, X_test, Y_test,timeindex,gp)

        # 
        # # # Different parameterization 
#         gp = gaussian_process.GaussianProcess(corr='cubic', theta0=1e-2, thetaL=1e-4, thetaU=1e-1,random_start=100)
#         Y_hat_test, sigma = my_gpfit(X_train,Y_train, X_test, Y_test,timeindex,gp)


        # Score_train = linregmodel1.score(X_train,Y_train)
        # Score_test = linregmodel1.score(X_test,Y_test)
        # RMSE_train = float((((Y_hat_train-Y_train)**2).mean())**(0.5))
        # RMSE_test = float((((Y_hat_test-Y_test)**2).mean())**(0.5))
        # result = [[Score_train, Score_test], [RMSE_train, RMSE_test]]

        Observed =  sum(Y_test)
        Expected = sum(Y_hat_test)
        sigma_joint = (sum(sigma))
        # # sigma_joint = (sum(sigma**2))**0.5
        Z[i] = (Observed-Expected)/sigma_joint

    topcomplaints_df['Z-Score'] = pd.DataFrame(Z)
    return topcomplaints_df
    

# Function to plot heatmaps using hist2d
# def my_heatmap(loc):
#     lat = loc['latitude'].values
#     lon = loc['longitude'].values
#     x=lon
#     y=lat
#     from matplotlib.colors import LogNorm
#     # plt.hist2d(x, y, bins=80, cmap='Greys', norm=LogNorm())
#     plt.hist2d(x, y, bins=100, norm=LogNorm())
#     plt.show()

# Function to plot heatmaps using hexplot    
# def my_heatmap2(loc):
#     lat = loc['latitude'].values
#     lon = loc['longitude'].values
#     x=lon
#     y=lat
#     from matplotlib.colors import LogNorm
#     plt.hexbin(x, y, gridsize=80,norm=LogNorm())
#     plt.show()        