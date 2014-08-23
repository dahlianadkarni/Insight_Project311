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

# from sklearn import preprocessing, linear_model
# from sklearn.feature_extraction import DictVectorizer


db_test = mdb.connect(user="root", host="localhost", passwd='dahlia',  db="db311_recent_2014", charset='utf8')

@app.route('/')
@app.route('/index')
def index():
    return render_template('index_homepage_dahlia.html')   


@app.route("/result")
def complaints_json_echo():
    borough = request.args.get('borough')
    period = int(request.args.get('period'))
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
    topcomplaints = []
    for i, result in enumerate(complaints_df.to_records(index=False)):
        if i > 10:
            break
        topcomplaints.append(dict(complaint_type=result[0], counts=result[1]))

#         query = "SELECT created_date, complaint_type FROM " + dbtable + " WHERE borough = '" + borough + "' OR city = '" + borough + "';"
#         complaints_df = psql.read_sql(query, parse_dates=['created_date'], con=db_test)
#     topcomplaints1 = complaints_df['complaint_type'][complaints_df['created_date']>=period_str].value_counts()[:5]
#     i=0
#     while i < len(topcomplaints1):
#         if i > 10:
#             break
#         topcomplaints.append(dict(complaint_type=topcomplaints1.index[i], counts=topcomplaints1[i], increase = '6% more than expected'))
#         i+=1      

    print borough, period, topcomplaints

    return render_template("results_complaints.html", topcomplaints=topcomplaints,period_str = period_str)
#     return render_template('predict_homepage.html') 
    
    
@app.route("/predict")
def predict_home():
    return render_template('predict_homepage.html') 
# 
# 
# @app.route("/predictresults")
# def predict_cv():
#     borough_predict = request.args.get('borough_predict')
#     complaint_type = request.args.get('complaint_type')
#     print borough_predict
#     with db:
#         cur = db.cursor()
#         query = "SELECT created_date, complaint_type FROM complaints_recent WHERE borough = '" + borough_predict + "' AND complaint_type = '" + complaint_type + "' LIMIT 5;"
#         complaints_df = psql.read_sql(query, con=db)
#     topcomplaints = []
#     complaints = []
#     for i, result in enumerate(complaints_df.to_records(index=False)):
#         if i > 10:
#             break
#         topcomplaints.append(dict(created_date=result[0], complaint_type=result[1]))
#     return render_template("results_predict.html", complaints=complaints)


@app.route('/visualize')
def visualize():
    return render_template('homepage_visualize.html')   



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

def my_heatmap_vis(loc):
    lat = loc['latitude'].values
    lon = loc['longitude'].values
    x=lon
    y=lat
    # plt.hist2d(x, y, bins=80, cmap='Greys', norm=LogNorm())
    plt.hist2d(x, y, bins=100, norm=LogNorm())
    plt.show()
    io = StringIO()
    plt.savefig(io, format='png', dpi=150)
    return io.getvalue().encode('base64')

        
@app.route("/visualize_result")
def visualize_result():
    complaint_type = request.args.get('complaint_type')
    period = int(request.args.get('period'))
    if period == 1:
        period_str = '2014-07-01'
        dbtable = 'complaints_201407'
    else:
        period_str = '2014-05-01'
        dbtable = 'complaints_recent'
    with db_test:
        query = "SELECT latitude,longitude FROM " + dbtable + " WHERE complaint_type = '" + complaint_type + "';"
        complaints_df = psql.read_sql(query, con=db_test)
    loc = complaints_df[['latitude','longitude']].dropna()
    print loc.head()
    return render_template("results_visualize.html", image = my_heatmap_vis(loc))
                
                
                
                
if __name__ == "__main__":
    app.run(host='0.0.0.0', port = 5000) 
#     app.run(host='0.0.0.0') 