from flask import Flask,render_template,jsonify,request
from app import app
import pymysql as mdb
import pandas.io.sql as psql
from cStringIO import StringIO
import matplotlib.pyplot as plt

db = mdb.connect(user="root", host="localhost", db="311db_till2013", charset='utf8')

@app.route('/')
@app.route('/index')
def index():
    return render_template('index_js_my_new.html')   

@app.route("/predict")
def predict_cv():
    return render_template('index_js_my_new2.html')   


@app.route('/port')
def port():
    return render_template('portfolio_manu.html'),
		
# def print_result(ComplaintTypeCount):
#     ComplaintTypeCount[:10].plot(kind='bar')
#     # plt.subplots(1, 1, sharex=True, figsize=(8, 6))
# #     for y in set(df.discharge_year):
# #         ax = np.log(df.total_charges[df.discharge_year==y]).plot(kind='kde',
# #                                                                  legend=True,
# #                                                                  label=y)
# #     ax.set_xlabel('Log-Total Charges')
# 
#     io = StringIO()
#     plt.savefig(io, format='png', dpi=150)
#     return io.getvalue().encode('base64')

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

def predict_image(complaints_volume):

# #     topcomplaintscount_df[:10].plot(kind='barh',legend=False)

#     f_year = complaints_volume.index.year
#     f_quarter = complaints_volume.index.quarter
#     f_month = complaints_volume.index.month
#     f_weekno = complaints_volume.index.weekofyear
#     f_dayofyr = complaints_volume.index.dayofyear
#     f_weekday = complaints_volume.index.weekday
#     f_day = complaints_volume.index.day
#     # f_hour = pd.DatetimeIndex(CreatedDate).hour
#     # f_min = pd.DatetimeIndex(CreatedDate).minute
#     g_year = pd.get_dummies(f_year)
#     g_quarter= pd.get_dummies(f_quarter)
#     g_month = pd.get_dummies(f_month)
#     g_weekno = pd.get_dummies(f_weekno)
#     g_dayofyr = pd.get_dummies(f_dayofyr)
#     g_weekday = pd.get_dummies(f_weekday)
#     g_day = pd.get_dummies(f_day)
# 
#     # feature_full = pd.concat([g_year,g_quarter,g_month,g_weekday,g_day], axis=1)  
#     # feature_full = pd.concat([g_year,g_quarter,g_month,g_weekno,g_dayofyr,g_weekday,g_day], axis=1)  
#     feature_full = pd.concat([g_year,g_quarter,g_month,g_weekno,g_dayofyr,g_weekday,g_day], axis=1)  
# 
#     feature_full.applymap(np.int) 
# 
#     # lenth_training = len(complaints_volume.index[complaints_volume.index<'2013-10-01'])
#     lenth_training=len(complaints_volume.index[complaints_volume.index<'2013-07-01'])
#     X_train = feature_full[:lenth_training]
#     X_test = feature_full[lenth_training:]
#     Y_train = complaints_volume[:lenth_training]
#     Y_test = complaints_volume[lenth_training:]
#     linregmodel = linear_model.LinearRegression()
#     # linregmodel = linear_model.Ridge (alpha = .1)
#     # linregmodel = linear_model.Lasso(alpha = 0.1)
#     linregmodel.fit(X_train,Y_train)
#     Ytrainpredicted = linregmodel.predict(X_train)
#     Ytestpredicted = linregmodel.predict(X_test)
#     # linregmodel.score(X_train,Y_train)
# #     plot(complaints_volume.index[lenth_training:],Y_test,'b',complaints_volume.index[lenth_training:],Ytestpredicted,'r')
#     plot(complaints_volume.index,complaints_volume,'b',complaints_volume.index[lenth_training:],Ytestpredicted,'r')

    complaints_volume.plot()
    io = StringIO()
    plt.savefig(io, format='png', dpi=150)
    return io.getvalue().encode('base64')
    
@app.route("/db_json_echo",methods=['GET'])
def complaints_json_echo():
    ret_data = {'borough': request.args.get('borough'),
                'period':  request.args.get('period')}
    with db:
        cur = db.cursor()
        query = "SELECT complaint_type,count(*) AS counts FROM complaints WHERE borough = '" + ret_data['borough'] + "' AND created_date > '10/01/2013' GROUP BY complaint_type ORDER BY counts DESC  LIMIT 5;"
        complaints_df = psql.read_sql(query, con=db)
    topcomplaints = []
    for i, result in enumerate(complaints_df.to_records(index=False)):
        if i > 10:
            break
        topcomplaints.append(dict(complaint_type=result[0], counts=result[1]))
    return jsonify(dict(topcomplaints=topcomplaints, image=print_hist(complaints_df.set_index('complaint_type'))))	
        
  	
@app.route("/db_json_echo2",methods=['GET'])
def complaints_json_echo2():
#     ret_data = {'borough': request.args.get('borough'),
#                 'period':  request.args.get('period')}
    ret_data = {'borough': request.args.get('borough'),
                'period':  request.args.get('period'),
                'complaint_type':  request.args.get('complaint_type'),
                'resolution':  request.args.get('resolution')           
                }
#     with db:
#         cur = db.cursor()
#         query = "SELECT complaint_type,count(*) AS counts FROM complaints WHERE borough = '" + ret_data['borough'] + "' AND created_date > '10/01/2013' GROUP BY complaint_type ORDER BY counts DESC  LIMIT 5;"
#         complaints_df = psql.read_sql(query, con=db)
#     topcomplaints = []
#     for i, result in enumerate(complaints_df.to_records(index=False)):
#         if i > 10:
#             break
#         topcomplaints.append(dict(complaint_type=result[0], counts=result[1]))
#     return jsonify(dict(topcomplaints=topcomplaints, image=print_hist(complaints_df.set_index('complaint_type'))))	

    with db:
        cur = db.cursor()
#         query = "SELECT created_date, complaint_type FROM complaints WHERE borough = '" + ret_data['borough'] + "' AND complaint_type = '" + ret_data['complaint_type'] + "' AND created_date > '10/01/2013';"
        query = "SELECT created_date, complaint_type FROM complaints WHERE created_date > '10/01/2013';"
        complaints_df = psql.read_sql(query, con=db)
#         complaints_volume_weekly= complaints.set_index('created_date').sort_index().resample('W', how=len)
#         complaints_volume_daily= complaints.set_index('created_date').sort_index().resample('D', how=len)
#         complaints_volume = complaints_df.set_index('created_date').sort_index().resample('resolution', how=len)
#         complaints_volume = complaints_df.set_index('created_date').sort_index().resample(ret_data['resolution'], how=len)
        complaints_volume = complaints_df.set_index('created_date').sort_index().resample('D', how=len)
    return jsonify(dict(image=predict_image(complaints_df.set_index('complaints_volume'))))	

 