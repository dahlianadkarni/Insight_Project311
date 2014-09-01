from flask import Flask
app = Flask(__name__)

from flask import render_template,jsonify,request
import pymysql as mdb
import matplotlib.pyplot as plt
import pandas as pd
import pandas.io.sql as psql
import numpy as np
from cStringIO import StringIO
from matplotlib.colors import LogNorm
import complaint311 as cp
import math

from sklearn import preprocessing, linear_model


#Homepage
@app.route('/')
@app.route('/index')
def index():
    return render_template('index_homepage_dahlia.html')   

#Result of homepage query - ranked list of complaints
@app.route("/result")
def result():
    period = int(request.args.get('period'))
    borough = request.args.get('borough')
    borough = borough.upper()
    cbn = request.args.get('cb')
    res = 'M'
#     
    mydict = {'BRONX': range(1,13),'BROOKLYN': range(1,19), 'MANHATTAN':range(1,13), 'QUEENS': range(1,15), 'STATEN ISLAND': range(1,4)}
    if int(cbn) in mydict[borough]:
        if len(cbn)==1:
            cb = "0"+cbn+" "+borough
        else:
            cb = cbn+" "+borough
        topcomplaints, pathfig = cp.get_zscore(cb,np.int(period),res,'cb',True) # # Last option is to save the fig
    else:
        topcomplaints, pathfig = cp.get_zscore(borough,np.int(period),res,'borough',True) # # Last option is to save the fig

#     pathfig = "static/trend.png"

    return render_template("results_complaints.html", topcomplaints=topcomplaints)
#     return render_template("results_complaints.html", topcomplaints=topcomplaints, pathfig = pathfig)
    

# Visualize homepage
@app.route('/visualize')
def visualize():
    return render_template('homepage_visualize.html')   

# Heatmap for a given complaint
def my_heatmap_vis(loc):
    lat = loc['latitude'].values
    lon = loc['longitude'].values
    x=lon
    y=lat
    # plt.hist2d(x, y, bins=80, cmap='Greys', norm=LogNorm())
#     plt.hist2d(x, y, bins=100, norm=LogNorm())
    plt.hexbin(x, y, gridsize=80,norm=LogNorm())
    path = "static/tempfig.png"
    plt.savefig(path, dpi=2500)
    return path

# result of visualization
@app.route("/visualize_result")
def visualize_result():
    complaint_type = request.args.get('complaint_type')
#     period = int(request.args.get('period'))
#     if period == 1:
#         period_str = 'in the last month'
#         dbtable = 'complaints_201407'
#     else:
#         period_str = 'in last 3 months'
#         dbtable = 'complaints_recent'
#     print dbtable
#     print period
#     with db_test:
#         query = "SELECT latitude,longitude FROM " + dbtable + " WHERE complaint_type = '" + complaint_type + "';"
#         complaints_df = psql.read_sql(query, con=db_test)
#     loc = complaints_df[['latitude','longitude']].dropna()
#     print loc.head()
#     path = my_heatmap_vis(loc)
#     image = path
    image = "static/"+complaint_type+".png"
    complaint_type.replace(' ','_')
#     print complaint_type
    return render_template("results_visualize.html",image = image, complaint_type = complaint_type)


def print_hist(topcomplaintscount_df):
    topcomplaintscount_df[:10].plot(kind='barh',legend=False)
    # plt.subplots(1, 1, sharex=True, figsize=(8, 6))
#     for y in set(df.discharge_year):
#         ax = np.log(df.total_charges[df.discharge_year==y]).plot(kind='kde',
#                                                                  legend=True,
#                                                                  label=y)
#     ax.set_xlabel('Log-Total Charges')

    io = StringIO()
    plt.savefig(io, format='png', dpi=150)
    return io.getvalue().encode('base64')
    
    
# # # Predict functions
# @app.route("/predict")
# def predict_home():
#     return render_template('predict_homepage.html') 
# 
# 
# @app.route("/predict_results")
# def predict_cv():
#     borough = request.args.get('borough')
#     borough = borough.upper()
#     cbn = request.args.get('cb')
#     res = 'D'
#     complaint_type = request.args.get('complaint_type')
# 
# #     
#     mydict = {'BRONX': range(1,13),'BROOKLYN': range(1,19), 'MANHATTAN':range(1,13), 'QUEENS': range(1,15), 'STATEN ISLAND': range(1,4)}
#     if int(cbn) in mydict[borough]:
#         if len(cbn)==1:
#             cb = "0"+cbn+" "+borough
#         else:
#             cb = cbn+" "+borough
#         locationstring = "community_board = '" + cb + "'"
#     else:
#         locationstring = "borough = '" + location + "'"
# 
#     db_all = mdb.connect(host='localhost',user='root',passwd='dahlia',db='db311_all_sincemay10')
#     with db_all:
#         query = "SELECT created_date, complaint_type FROM complaints WHERE created_date IS NOT NULL AND " +locationstring + " AND complaint_type = '" + complaint_type + "' ;"
#         cv = psql.read_sql(query, parse_dates=['created_date'], con=db_all).set_index('created_date')          
#     cv = cv.sort_index().resample(res,how=len)
#     
#     L = 14
#     newdays = cv.index[-L:]+L
#     X = extract_features(cv.index+newdays,res)
#     X_train = X[:-L]
#     X_test = X[-L:]
#     Y_train = cv.values
#             
#        
#     # # #  --------LINEAR REGRESSION-------
#     # # If you want to infer alpha for Ridge regression via cross validation
#     linregmodel0 = linear_model.RidgeCV (alphas=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0])
#     linregmodel0.fit(X_train,Y_train)
#     alp = linregmodel0.alpha_ 
#     # alp = 0.5
#     #
#      # Linear regression - OLS or Ridge or Lasso
#     linregmodel = linear_model.LinearRegression()
# #         linregmodel = linear_model.Ridge (alpha = alp)
# #         linregmodel = linear_model.Lasso(alpha = 10)
# 
# #         Y_hat_test, sigma = my_linfit(X_train,Y_train, X_test,linregmodel)
# 
#     
#     # # #  --------GAUSSIAN PROCESS -----
#     # # Gaussian Processes
#     gp = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
#     Y_hat_test, sigma = my_gpfit(X_train,Y_train, X_test,gp)
# 
#     # # # Optional noise feature
#     # gp = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
#     # y=Y_train
#     # dy = 0.5 + 1.0 * np.random.random(y.shape)
#     # noise = np.random.normal(0, dy)
#     # y += noise
#     # Y_hat_test, sigma = my_gpfit(X_train,y, X_test,gp)
#     # 
#     # 
#     # # # Different parameterization 
#     # gp = gaussian_process.GaussianProcess(corr='cubic', theta0=1e-2, thetaL=1e-4, thetaU=1e-1,random_start=100)
#     # Y_hat_test, sigma = my_gpfit(X_train,Y_train, X_test,gp)
# 
#     # # Scoring
#     # Score_train = linregmodel1.score(X_train,Y_train)
#     # Score_test = linregmodel1.score(X_test,Y_test)
#     # RMSE_train = float((((Y_hat_train-Y_train)**2).mean())**(0.5))
#     # RMSE_test = float((((Y_hat_test-Y_test)**2).mean())**(0.5))
#     # result = [[Score_train, Score_test], [RMSE_train, RMSE_test]]
# 
# 
#     return render_template("results_predict.html", complaints=complaints)

       
if __name__ == "__main__":
    app.run(host='0.0.0.0', port = 5000) 
#     app.run(host='0.0.0.0') 
