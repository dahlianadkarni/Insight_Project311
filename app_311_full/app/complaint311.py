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
import math
from cStringIO import StringIO

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

        # # # Normalize linear features
#         f_year = (f_year-mean(f_year))/std(f_year)+0.5
#         f_quarter = (f_quarter-mean(f_quarter))/std(f_quarter)+0.5
#         f_month = (f_month-mean(f_month))/std(f_month)+0.5
#         f_weekno = (f_weekno-mean(f_weekno))/std(f_weekno)+0.5

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

# Linear regression fit   
def my_linfit(X_train,Y_train, X_test, linregmodel1):
    linregmodel1.fit(X_train,Y_train)
    # cf = linregmodel.coef_
    Y_hat_train = linregmodel1.predict(X_train)
    Y_hat_test = linregmodel1.predict(X_test)

    n=len(Y_train)
    sigma_hat =float(((((Y_hat_train-Y_train)**2).sum())/(n-2))**0.5)
    sigma = sigma_hat*(((X_test.values-X_train.mean().values)**2).sum(axis=1)/((X_train.values-X_train.mean().values)**2).sum(axis=1).sum(axis=0)+1/n+1)**0.5
    
    return Y_hat_test, sigma

# Gaussian processes fit 
def my_gpfit(X_train,Y_train, X_test, gp):
    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(X_train,Y_train) 

    # Make the prediction (ask for MSE as well)
    Y_hat_test, MSE = gp.predict(X_test, eval_MSE=True)
    sigma = np.sqrt(MSE)

    return Y_hat_test, sigma


# Find out the top complaints (in terms of volume) for a given borough and period
def topcomplaint(locationstring,period):
    startmonth = "0"+str(9-period)
    db_recent = mdb.connect(host='localhost',user='root',passwd='dahlia',db='db311_recent_03to08')
    with db_recent:
        query = "SELECT complaint_type,count(*) AS counts FROM complaints_recent WHERE created_date IS NOT NULL AND created_date > '" +startmonth+ "' AND " +locationstring + " GROUP BY complaint_type ORDER BY counts DESC  LIMIT 10;"
        topcomplaints_df = psql.read_sql(query, con=db_recent)
    return topcomplaints_df
    
# function that takes a Borough/CB, period and outputs a list of top complaints by priority(z-score)    
def get_zscore(location,period,res,borough_or_cb,printfig):

    if borough_or_cb == 'borough':
        locationstring = "borough = '" + location + "'"
        borough = location
    else:
        locationstring = "community_board = '" + location + "'"
        cb = location

    db_all = mdb.connect(host='localhost',user='root',passwd='dahlia',db='db311_all_sincemay10')

    
    startTime = datetime.utcnow()
    topcomplaints_df = topcomplaint(locationstring,period)    
    topcomplaints = topcomplaints_df['complaint_type']
    listtopcomplaints = list(topcomplaints_df['complaint_type'])
    print (datetime.utcnow()-startTime)        
    mystring = "('" + "', '".join(listtopcomplaints) + "')"
    TopN = len(listtopcomplaints) 
    Z = np.zeros(TopN)
        
    for i in range(TopN):
    
        with db_all:
            query = "SELECT created_date, complaint_type FROM complaints WHERE created_date IS NOT NULL AND " +locationstring + " AND complaint_type = '" + topcomplaints[i] + "' ;"
            cv = psql.read_sql(query, parse_dates=['created_date'], con=db_all).set_index('created_date')          
        cv = cv.sort_index().resample(res,how=len)
          
        if len(cv[(cv.index > '2011-05-01') & (cv.index < '2013-05-01')])<5:
            Z[i]=float('NaN')
            continue

        L = period
        X = extract_features(cv.index,res)
        X_train = X[:-L]
        X_test = X[-L:]
        Y_train = cv.values[:-L]
        Y_test = cv.values[-L:]
                
        
        
        # # #  --------LINEAR REGRESSION-------
        # # If you want to infer alpha for Ridge regression via cross validation
        linregmodel0 = linear_model.RidgeCV (alphas=[0.01,0.02, 0.04, 0.08,0.16,0.32, 0.64, 1.28,2.56, 5.12, 10.24])
        linregmodel0.fit(X_train,Y_train)
        alp = linregmodel0.alpha_ 
        # alp = 0.5
        #
         # Linear regression - OLS or Ridge or Lasso
#         linregmodel = linear_model.LinearRegression()
        linregmodel = linear_model.Ridge (alpha = alp)
#         linregmodel = linear_model.Lasso(alpha = 10)
 
        Y_hat_test, sigma = my_linfit(X_train,Y_train, X_test,linregmodel)

        
        # # #  --------GAUSSIAN PROCESS -----
        # # Gaussian Processes
#         gp = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
#         Y_hat_test, sigma = my_gpfit(X_train,Y_train, X_test,gp)

        # # # Optional noise feature
        # gp = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
        # y=Y_train
        # dy = 0.5 + 1.0 * np.random.random(y.shape)
        # noise = np.random.normal(0, dy)
        # y += noise
        # Y_hat_test, sigma = my_gpfit(X_train,y, X_test,gp)
        # 
        # 
        # # # Different parameterization 
#         gp = gaussian_process.GaussianProcess(corr='cubic', theta0=1e-2, thetaL=1e-4, thetaU=1e-1,random_start=100)
#         Y_hat_test, sigma = my_gpfit(X_train,Y_train, X_test,gp)

        # # Scoring
        # Score_train = linregmodel1.score(X_train,Y_train)
        # Score_test = linregmodel1.score(X_test,Y_test)
        # RMSE_train = float((((Y_hat_train-Y_train)**2).mean())**(0.5))
        # RMSE_test = float((((Y_hat_test-Y_test)**2).mean())**(0.5))
        # result = [[Score_train, Score_test], [RMSE_train, RMSE_test]]

        # # Get corresponding Z-scores
        Observed =  sum(Y_test)
        Expected = sum(Y_hat_test)
        sigma_joint = (sum(sigma))
        # # sigma_joint = (sum(sigma**2))**0.5
        Z[i] = (Observed-Expected)/sigma_joint
    topcomplaints_df['Z-Score'] = pd.DataFrame(Z)
    topcomplaints_df = topcomplaints_df.sort(['Z-Score'],ascending=False)    
    print (datetime.utcnow()-startTime)

    # # Convert to a dictionary to output
    topcomplaints = []
    for i, result in enumerate(topcomplaints_df.to_records(index=False)):
        if i >= 5 or math.isnan(result[2]):
            break
        if result[2]>3:
            pr = "Very High"
        elif result[2]>2:
            pr = "High"
        elif result[2]>1:
            pr="Medium"
        elif result[2]<0:
            pr = "Complaint volume less than expected"
        else:
            pr = "Low"        
        if i==0:
            temp = result[1]
            maxcomplaint = result[0]
        else:
            if result[1]>temp:
                maxcomplaint = result[0]   
                temp = result[1]     
        s = result[0][:1]+result[0][1:].lower()
        topcomplaints.append(dict(complaint_type=s, counts=result[1], priority = pr,zscore = '%0.2f' %(result[2])  ))
    
    # # If you want to print/save the figure
    if printfig:   

        fig_res = res 
        maxcomplaint   
        with db_all:
            query = "SELECT created_date, complaint_type FROM complaints WHERE created_date IS NOT NULL AND " +locationstring + " AND complaint_type = '" + maxcomplaint + "' ;"
            cv = psql.read_sql(query, parse_dates=['created_date'], con=db_all).set_index('created_date')          
        cv = cv.sort_index().resample(res,how=len) 
 
        maxcomplaint2 = topcomplaints[0]['complaint_type']
        with db_all:
            query = "SELECT created_date, complaint_type FROM complaints WHERE created_date IS NOT NULL AND " +locationstring + " AND complaint_type = '" + maxcomplaint2 + "' ;"
            cv2 = psql.read_sql(query, parse_dates=['created_date'], con=db_all).set_index('created_date')          
        cv2 = cv2.sort_index().resample(res,how=len)
        
        plt.close('all')
        plt.figure()
        plt.plot(cv.index,cv,label = maxcomplaint)
        plt.plot(cv2.index,cv2,label = maxcomplaint2)
        legend1 = plt.legend(loc='upper center', shadow=True)
        pathfig = "static/trend.png"
        plt.savefig(pathfig)
#         io = StringIO()
#         plt.savefig(io, format='png', dpi=100)
#         pathfig = io.getvalue().encode('base64')
    print (datetime.utcnow()-startTime)
    return topcomplaints, pathfig    
    
  
  
  
    

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