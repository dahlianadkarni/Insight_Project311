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
def complaints_json_echo():
    borough = request.args.get('borough')
    borough = borough.upper()
    print borough
    period = int(request.args.get('period'))
    if period == 1:
        period_str = 'in the last month'
        dbtable = 'complaints_201407'
    else:
        period_str = 'in last 3 months'
        dbtable = 'complaints_recent'

    complaints_df = cp.get_zscore(borough,np.int(period),'M')
    complaints_df = complaints_df.sort(['Z-Score'],ascending=False)    
    
    topcomplaints = []
    for i, result in enumerate(complaints_df.to_records(index=False)):
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
        
#         if i==0:
#             temp = result[1]
#             maxcomplaint = result[0]
#         else:
#             if result[1]>temp:
#                 maxcomplaint = result[0]        
        s = result[0][:1]+result[0][1:].lower()
        topcomplaints.append(dict(complaint_type=s, counts=result[1], priority = pr,zscore = '%0.2f' %(result[2])  ))
    
#     print maxcomplaint
#     complaint_here = maxcomplaint.replace('-','_').replace(' ','_').replace('\n','_').replace('*','').replace('(','').replace(')','').replace('/','_')
#     filenametest = "311PKLData/"+borough + "_test" +str(period)+"_"+ complaint_here + ".pkl"
#     cv_test = pd.read_pickle(filenametest)
#     filenametrain = "311PKLData/"+borough + "_train"+str(period)+"_" + complaint_here + ".pkl"
#     cv_train = pd.read_pickle(filenametrain)  
#     cv_test = cv_test.sort_index().resample('2W',how=sum)
#     cv_train = cv_train.sort_index().resample('2W',how=sum)
#     cv = pd.concat([cv_train,cv_test])
#     
#     plt.figure()
#     plt.plot(cv.index,cv,label = maxcomplaint)
#          
#     maxcomplaint = topcomplaints[0]['complaint_type']
#     print maxcomplaint
#     complaint_here = maxcomplaint.replace('-','_').replace(' ','_').replace('\n','_').replace('*','').replace('(','').replace(')','').replace('/','_')
#     filenametest = "311PKLData/"+borough + "_test" +str(period)+"_"+ complaint_here + ".pkl"
#     cv_test = pd.read_pickle(filenametest)
#     filenametrain = "311PKLData/"+borough + "_train"+str(period)+"_" + complaint_here + ".pkl"
#     cv_train = pd.read_pickle(filenametrain)  
#     cv_test = cv_test.sort_index().resample('2W',how=sum)
#     cv_train = cv_train.sort_index().resample('2W',how=sum)
#     cv = pd.concat([cv_train,cv_test])
#     
#     plt.plot(cv.index,cv,label = maxcomplaint)
#     legend1 = plt.legend(loc='upper center', shadow=True)
#     
#     pathfig = "static/trend.png"
#     plt.savefig(pathfig)
# 
    
    return render_template("results_complaints.html", topcomplaints=topcomplaints, period_str = period_str)

#     return render_template("results_complaints.html", topcomplaints=topcomplaints, period_str = period_str, pathfig = pathfig)
#     return render_template("results_complaints.html", topcomplaints=topcomplaints,topcomplaints1=topcomplaints1, period_str = period_str)
    

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
    
    
    # Predict functions
# @app.route("/predict")
# def predict_home():
#     return render_template('predict_homepage.html') 
# 
# 
# @app.route("/predict_results")
# def predict_cv():
#     borough_predict = request.args.get('borough_predict')
#     complaint_type = request.args.get('complaint_type')
#     table_test = 'complaints_recent'
#     table_train = 'complaints3'
#     db_train = db_train3
#     complaint_type = complaint_type.replace('-','_').replace(' ','_').replace('\n','_').replace('*','').replace('(','').replace(')','').replace('/','_')
#     filenametest = "311PKLData/"+borough_predict + "_test" +str(period)+"_"+ complaint_here + ".pkl"
#     cv_test = pd.read_pickle(filenametest)
#     filenametrain = "311PKLData/"+borough_predict + "_train"+str(period)+"_" + complaint_here + ".pkl"
#     cv_train = pd.read_pickle(filenametrain)  
#     res = W
#     cv_test = cv_test.sort_index().resample(res,how=sum)
#     cv_train = cv_train.sort_index().resample(res,how=sum)              
#     L = len(cv_test)
#     cv = pd.concat([cv_train,cv_test])
#     cv_train = cv
#     X = extract_features(cv.index,res)
#     X_train = X[:-L]
#     X_test = X[-L:]
#     Y_train = cv_train.values
#     Y_test = cv_test.values
#     timeindex = cv_test.index
#     if len(cv_train[(cv_train.index > '2011-05-01') & (cv_train.index < '2013-05-01')])<5:
#         error = "Not sufficient data"
#         print error
#         break
#         
#         # # If you want to infer alpha for Ridge regression via cross validation
# #         linregmodel0 = linear_model.RidgeCV (alphas=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0])
# #         linregmodel0.fit(X_train,Y_train)
# #         alp = linregmodel0.alpha_ 
# #         # alp = 0.5
# # 
#         # # Linear regression - OLS or Ridge or Lasso
# #         linregmodel = linear_model.LinearRegression()
# #         linregmodel = linear_model.Ridge (alpha = alp)
#         # linregmodel = linear_model.Lasso(alpha = 10)
# # 
# #         Z1[i] = my_linfit(X_train,Y_train, X_test, Y_test, cv_test.index,linregmodel)
# #          
#         # # Gaussian Processes
#         gp = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
#         # # # OPtional noise feature
# #         y=Y_train
# #         dy = 0.5 + 1.0 * np.random.random(y.shape)
# #         noise = np.random.normal(0, dy)
# #         y += noise
#    # Fit to data using Maximum Likelihood Estimation of the parameters
#     gp.fit(X_train,Y_train) 
# 
#     # Make the prediction (ask for MSE as well)
#     Y_hat_test, MSE = gp.predict(X_test, eval_MSE=True)
#     sigma = np.sqrt(MSE)
# 
#     # Score_train = linregmodel.score(X_train,Y_train)
#     # Score_test = linregmodel.score(X_test,Y_test)
#     # RMSE_train = float((((Y_hat_train-Y_train)**2).mean())**(0.5))
#     # RMSE_test = float((((Y_hat_test-Y_test)**2).mean())**(0.5))
#     
#     Observed =  sum(Y_test)
#     Expected = sum(Y_hat_test)
#     sigma_joint = (sum(sigma**2))**0.5
#     Z = (Observed-Expected)/sigma_joint
# 
#     plot(timeindex,Y_test,'b')
#     plot(timeindex,Y_hat_test,'r')
#     plot(timeindex,(Y_hat_test.T+1.96*sigma).T,'r:')
#     plot(timeindex,(Y_hat_test.T-1.96*sigma).T,'r:')
# 
#         
#         
# 
# 
# 
# 
#     return render_template("results_predict.html", complaints=complaints)
# 
       
if __name__ == "__main__":
    app.run(host='0.0.0.0', port = 5000) 
#     app.run(host='0.0.0.0') 
