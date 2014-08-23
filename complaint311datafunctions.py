#!/Users/dahlia/Documents/insight_virtual_env/bin/python
import sys
import pandas as pd
import numpy as np
import csv
import pymysql as mdb
import pandas.io.sql as psql

# import matplotlib.pyplot as plt
# pd.options.display.mpl_style = 'default'
# import datetime
# from sklearn import preprocessing, linear_model
# from sklearn.feature_extraction import DictVectorizer
# from scipy.spatial.distance import pdist, squareform
# from scipy.cluster.hierarchy import linkage, dendrogram
# 
# 
db_test = mdb.connect(host='localhost', user='root', passwd='dahlia', db='db311_recent_2014')
    
    
def extract_features(Xtimes,resolution):
    f_year = Xtimes.year
    f_quarter = Xtimes.quarter
    f_month = Xtimes.month
    f_weekno = Xtimes.weekofyear
    f_dayofyr = Xtimes.dayofyear
    f_weekday = Xtimes.weekday
    f_day = Xtimes.day
    g_year = pd.get_dummies(f_year)
    g_quarter= pd.get_dummies(f_quarter)
    g_month = pd.get_dummies(f_month)
    g_weekno = pd.get_dummies(f_weekno)
    g_dayofyr = pd.get_dummies(f_dayofyr)
    g_weekday = pd.get_dummies(f_weekday)
    g_day = pd.get_dummies(f_day)
#     if (f_weekday==0 or f_weekday==6):
#         weekend=1
#     else:
#         weekend=0
    # feature_full = pd.concat([g_year,g_quarter,g_month,g_weekday,g_day], axis=1)  
    # feature_full = pd.concat([g_year,g_quarter,g_month,g_weekno,g_dayofyr,g_weekday,g_day], axis=1)  
    if resolution =='D':
#         feature_full = pd.concat([weekend,f_year,f_quarter,f_month,f_weekno,f_dayofyear,f_weekday,f_day,g_year,g_quarter,g_month,g_weekno,g_dayofyr,g_weekday,g_day], axis=1)  
#         feature_full = pd.concat([f_year,f_quarter,f_month,f_weekno,f_dayofyear,f_weekday,f_day,g_year,g_quarter,g_month,g_weekno,g_dayofyr,g_weekday,g_day], axis=1)  
        feature_full = pd.concat([g_year,g_quarter,g_month,g_weekno,g_dayofyr,g_weekday,g_day], axis=1)  
    elif resolution == 'W':
#         feature_full = pd.concat([f_year,f_quarter,f_month,f_weekno,g_year,g_quarter,g_month,g_weekno], axis=1)  
        feature_full = pd.concat([g_year,g_quarter,g_month,g_weekno,g_weekday], axis=1)  
    elif resolution == 'M':
#         feature_full = pd.concat([f_year,f_quarter,f_month,g_year,g_quarter,g_month], axis=1)  
        feature_full = pd.concat([g_year,g_quarter,g_month], axis=1)  
    # feature_full = pd.concat([g_quarter,g_month,g_weekno,], axis=1)  
    feature_full.applymap(np.int) 
    return feature_full
    
def fit_linear_reg(X_train,Y_train,X_test,Y_test):
    linregmodel = linear_model.LinearRegression()
    # linregmodel = linear_model.Ridge (alpha = .1)
    # linregmodel = linear_model.Lasso(alpha = 0.1)
    
    linregmodel.fit(X_train,Y_train)
    Y_hat_train = linregmodel.predict(X_train)
    Y_hat_test = linregmodel.predict(X_test)
    Score_train = linregmodel.score(X_train,Y_train)
    Score_test = linregmodel.score(X_test,Y_test)
    
    RMSE_train = (((Y_hat_train-Y_train)**2).mean())**(0.5)
    RMSE_test = (((Y_hat_test-Y_test)**2).mean())**(0.5)
    
    return Y_hat_test, Y_hat_train, Score_test, Score_train, RMSE_test, RMSE_train

def my_heatmap(loc):
    lat = loc['latitude'].values
    lon = loc['longitude'].values
    x=lon
    y=lat
    from matplotlib.colors import LogNorm
    # plt.hist2d(x, y, bins=80, cmap='Greys', norm=LogNorm())
    plt.hist2d(x, y, bins=100, norm=LogNorm())
    plt.show()
    
def my_heatmap2(loc):
    lat = loc['latitude'].values
    lon = loc['longitude'].values
    x=lon
    y=lat
    from matplotlib.colors import LogNorm
    plt.hexbin(x, y, gridsize=80,norm=LogNorm())
    plt.show()    

def topcomplaint(borough,period):
    if period == 1:
        period_str = '2014-07-01'
        dbtable = 'complaints_201407'
    else:
        period_str = '2014-05-01'
        dbtable = 'complaints_recent'
    topcomplaints = []
    with db_test:
        query = "SELECT complaint_type,count(*) AS counts FROM " + dbtable + " WHERE borough = '" + borough + "' GROUP BY complaint_type ORDER BY counts DESC  LIMIT 5;"
        complaints_df = psql.read_sql(query, con=db_test)
#     print complaints_df['complaint_type'][0]
#     topcomplaints = []
#     for i, result in enumerate(complaints_df.to_records(index=False)):
#         if i > 10:
#             break
#         topcomplaints.append(dict(complaint_type=result[0], counts=result[1]))
#     print topcomplaints
#     print type(topcomplaints)
    return complaints_df['complaint_type'][0]
    
def complaint_times(borough,period,complaint_type):
    if period == 1:
        period_str = '2014-07-01'
        dbtable = 'complaints_201407'
    else:
        period_str = '2014-05-01'
        dbtable = 'complaints_recent'
    topcomplaints = []
    with db_test:
        query = "SELECT created_date, complaint_type FROM " + dbtable + " WHERE complaint_type = '" +complaint_type+ "' AND borough = '" + borough + "' OR city = '" + borough + "';"
        complaints_df = psql.read_sql(query, parse_dates=['created_date'], con=db_test)
    complaints_volume_weekly= complaints_df.set_index('created_date').sort_index().resample('W', how=len)    
    print complaints_volume_weekly.head()
    return    


# Gather our code in a main() function
def main():
    # Command line args are in sys.argv[1], sys.argv[2] ...
    # sys.argv[0] is the script name itself and can be ignored
    if len(sys.argv) != 3:
        print 'usage:  ./complaints311datafunctions.py borough period'
        sys.exit(1)

    borough = sys.argv[1]
    period = sys.argv[2]
    topcomplaint_here = topcomplaint(borough,period)
    print borough, period, topcomplaint_here
    complaint_times(borough,period,topcomplaint_here)
    
    
# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
    main()