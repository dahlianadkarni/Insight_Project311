
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np
import csv
import pymysql as mdb
import pandas.io.sql as psql
import matplotlib.pyplot as plt

# pd.options.display.mpl_style = 'default'
# import datetime
from sklearn import preprocessing, linear_model
from sklearn.feature_extraction import DictVectorizer
# import scipy.stats
# import scipy
# import math

mydb = mdb.connect(host='localhost',
    user='root',
    passwd='',
    db='311db_till2013')


# In[3]:

# topagency = psql.read_sql("SELECT agency, count(*) FROM complaints GROUP BY agency ORDER BY COUNT(*) DESC", con=mydb)
# topagency.head()


# In[10]:

# topcomplaintsHPD = psql.read_sql("SELECT complaint_type, count(*) FROM complaints WHERE agency='HPD'  GROUP BY complaint_type ORDER BY COUNT(*) DESC", con=mydb)
# topcomplaintsHPD.head()

# topcomplaintsHPD = psql.read_sql("SELECT complaint_type, count(*) FROM complaints WHERE agency='DOT'  GROUP BY complaint_type ORDER BY COUNT(*) DESC", con=mydb)
# topcomplaintsHPD.head()

# topcomplaintsHPD = psql.read_sql("SELECT agency,complaint_type, count(*) FROM complaints GROUP BY agency,complaint_type ORDER BY COUNT(*) DESC", con=mydb)
# topcomplaintsHPD.head()

topcomplaintsHPD = psql.read_sql("SELECT complaint_type,city, borough, count(*) FROM complaints GROUP BY city,borough,complaint_type ORDER BY COUNT(*) DESC", con=mydb)
topcomplaintsHPD.head()


# In[5]:

# # topcomplaintsHPD = psql.read_sql("SELECT complaint_type, incident_zip, count(*) FROM complaints WHERE closed_date IS NOT NULL AND agency='HPD'  GROUP BY complaint_type,incident_zip ORDER BY COUNT(*) DESC", con=mydb)
# # topcomplaintsHPD.head()

# # topcomplaintsHPD = psql.read_sql("SELECT complaint_type, descriptor, count(*) FROM complaints WHERE closed_date IS NOT NULL AND agency='HPD'  GROUP BY complaint_type,descriptor ORDER BY COUNT(*) DESC", con=mydb)
# # topcomplaintsHPD.head()

# # topcomplaintsRodent = psql.read_sql("SELECT complaint_type,descriptor, incident_zip, count(*) FROM complaints WHERE complaint_type='Rodent' GROUP BY descriptor,incident_zip ORDER BY COUNT(*) DESC", con=mydb)
# # topcomplaintsRodent.head()

# topcomplaintsNoise = psql.read_sql("SELECT complaint_type,descriptor,community_board, incident_zip, borough, count(*) FROM complaints WHERE complaint_type='Noise - Street/Sidewalk' GROUP BY borough ORDER BY COUNT(*) DESC", con=mydb)
# topcomplaintsNoise.head()


# In[6]:

# # #agency='HPD' AND complaint_type!='HEATING'
# topcomplaintsHPD = psql.read_sql("SELECT incident_zip,complaint_type, count(*) FROM complaints WHERE agency='HPD' AND complaint_type!='HEATING' AND closed_date IS NOT NULL and due_date IS NOT NULL GROUP BY complaint_type,incident_zip ORDER BY COUNT(*) DESC", con=mydb)

# # #complaint_type = 'HEATING'
# topcomplaintsHPD = psql.read_sql("SELECT complaint_type, descriptor, count(*) FROM complaints WHERE closed_date IS NOT NULL AND complaint_type='HEATING' GROUP BY complaint_type,descriptor ORDER BY COUNT(*) DESC", con=mydb)

# # # # complaint_type='GENERAL CONSTRUCTION' AND incident_zip=11226
# topcomplaintsHPD = psql.read_sql("SELECT incident_zip,complaint_type,descriptor, count(*) FROM complaints WHERE complaint_type='GENERAL CONSTRUCTION' AND incident_zip=11226 AND closed_date IS NOT NULL GROUP BY descriptor ORDER BY COUNT(*) DESC", con=mydb)

# topcomplaintsHPD.head()


# In[3]:

# # # complaint_type='GENERAL CONSTRUCTION' AND incident_zip=11226
# dataclosed = psql.read_sql("SELECT created_date,closed_date,descriptor,due_date,status,latitude,longitude FROM complaints WHERE closed_date IS NOT NULL AND created_date IS NOT NULL AND complaint_type='GENERAL CONSTRUCTION' AND incident_zip=11226", parse_dates=['created_date','closed_date'], con=mydb)

# # # descriptor='MOLD'
# dataclosed = psql.read_sql("SELECT created_date, closed_date, agency, incident_zip, complaint_type, descriptor, due_date FROM complaints WHERE closed_date IS NOT NULL AND created_date IS NOT NULL AND descriptor='MOLD'", parse_dates=['created_date','closed_date'], con=mydb)

# # # descriptor='Street%'
# dataclosed = psql.read_sql("SELECT created_date, closed_date, agency, incident_zip, complaint_type, descriptor, due_date FROM complaints WHERE closed_date IS NOT NULL AND created_date IS NOT NULL AND complaint_type LIKE 'Street%'", parse_dates=['created_date','closed_date'], con=mydb)

# # # complaint_type='Street%'
# dataclosed = psql.read_sql("SELECT created_date, closed_date, agency, incident_zip, complaint_type, descriptor, due_date FROM complaints WHERE closed_date IS NOT NULL AND created_date IS NOT NULL AND complaint_type='Street Light Condition'", parse_dates=['created_date','closed_date'], con=mydb)

# # complaint_type='Rodent%'
# dataclosed = psql.read_sql("SELECT created_date, closed_date, agency, incident_zip, complaint_type, descriptor, due_date FROM complaints WHERE closed_date IS NOT NULL AND created_date IS NOT NULL AND complaint_type='Rodent'", parse_dates=['created_date','closed_date'], con=mydb)

# # complaint_type='Noise - Street/Sidewalk' AND manhattn
# dataclosed = psql.read_sql("SELECT created_date, agency, complaint_type, descriptor,community_board, borough,incident_zip,closed_date FROM complaints WHERE created_date IS NOT NULL AND complaint_type='Noise - Street/Sidewalk' AND Borough='MANHATTAN'", parse_dates=['created_date','closed_date'], con=mydb)

# # # descriptor='Heat', Bronx
dataclosed = psql.read_sql("SELECT created_date, closed_date, agency, incident_zip, complaint_type, descriptor, due_date FROM complaints WHERE closed_date IS NOT NULL AND created_date IS NOT NULL AND descriptor='HEAT' AND (borough='BRONX' OR city = 'BRONX')", parse_dates=['created_date','closed_date'], con=mydb)
len(dataclosed)
# # complaint_type='Noise'
# dataclosed = psql.read_sql("SELECT created_date, closed_date, agency, incident_zip, complaint_type, descriptor, due_date FROM complaints WHERE closed_date IS NOT NULL AND created_date IS NOT NULL AND complaint_type LIKE 'Noise%'", parse_dates=['created_date','closed_date'], con=mydb)

# dataclosed.head()


# In[12]:

# set(dataclosed['descriptor'])
# dataclosed = dataclosed[dataclosed['descriptor']=='HEAT']
# len(dataclosed[dataclosed['due_date'] is None])
# dataclosed['due_date'][1] is None
# len(dataclosed)
# len(set(dataclosed['community_board']))
# len(set(dataclosed['incident_zip']))


# In[22]:

# complaints = dataclosed[['created_date','descriptor']]
index = (dataclosed['created_date']>'2010-01-01') & (dataclosed['created_date']<'2015-01-01')
dataclosed1 = dataclosed[index]
complaints = dataclosed1[['created_date','complaint_type']]
len(complaints)
complaints_volume_weekly= complaints.set_index('created_date').sort_index().resample('W', how=len)
complaints_volume_daily= complaints.set_index('created_date').sort_index().resample('D', how=len)
complaints_volume = complaints_volume_daily
# len(complaints_volume)
complaints_volume.plot()
# len(complaints_volume)
# complaints_volume.plot()


# In[23]:

ClosedDate = pd.to_datetime(pd.Series(dataclosed['closed_date']))
CreatedDate = pd.to_datetime(pd.Series(dataclosed['created_date']))
ActualTime = ClosedDate-CreatedDate
ActualTime = ActualTime.astype('timedelta64[h]')

index0 = (ActualTime>0)
# index2 = (dataclosed['created_date']>'2013-01-01')
index1 = index0 #& index2

ActualTime = ActualTime[index1]
dataclosed1 = dataclosed[index1]
dataclosed1['actual_time']=ActualTime


ResTimesAct = dataclosed1[['created_date','actual_time']].set_index('created_date').sort_index().resample('D', how='mean')
ResTimesAct.plot()
len(ResTimesAct)

# complaints = dataclosed[['created_date','descriptor']][index1]
# complaints_volume= complaints.set_index('created_date').sort_index().resample('W', how=len)
complaints_volume.plot()
len(complaints_volume)

# ResTimesAct['ResTimeActual']
# ResTimesAct = pd.DataFrame(ResTimesAct)
# ResTimesAct = ResTimesAct.set_index('Created Date').sort_index()
# ResTimesAct
# # ResTimesAct.astype('timedelta64[m]').mean()
# ResTimesAct.astype('timedelta64[m]').std()
# ResTimesAct.astype('timedelta64[m]').resample('D', how='mean').plot()


# In[64]:

# plot(ResTimesAct,log(complaints_volume),'.')
# # plot(log(ResTimesAct),log(complaints_volume),'.')
len(ResTimesAct)
len(complaints_volume)
# plot(ResTimesAct,complaints_volume,'.')
type(complaints_volume)
# complaints_volume.index
# complaints_volume.index.year
# len(complaints_volume.index)
len(complaints_volume.index[complaints_volume.index<'2012-07-01'])
# len(complaints_volume.index[complaints_volume.index>'2012-07-01'])


# In[128]:

# # CreatedDate = dataclosed['created_date'][:10]
# CreatedDate = dataclosed['created_date']
# f_year = pd.DatetimeIndex(CreatedDate).year
# f_quarter = pd.DatetimeIndex(CreatedDate).quarter
# f_month = pd.DatetimeIndex(CreatedDate).month
# f_weekno = pd.DatetimeIndex(CreatedDate).weekofyear
# # f_dayofyr = pd.DatetimeIndex(CreatedDate).dayofyear
# f_weekday = pd.DatetimeIndex(CreatedDate).weekday
# f_day = pd.DatetimeIndex(CreatedDate).day
# # f_hour = pd.DatetimeIndex(CreatedDate).hour
# # f_min = pd.DatetimeIndex(CreatedDate).minute
# # len(f_year)


f_year = complaints_volume.index.year
f_quarter = complaints_volume.index.quarter
f_month = complaints_volume.index.month
f_weekno = complaints_volume.index.weekofyear
f_dayofyr = complaints_volume.index.dayofyear
f_weekday = complaints_volume.index.weekday
f_day = complaints_volume.index.day
# f_hour = pd.DatetimeIndex(CreatedDate).hour
# f_min = pd.DatetimeIndex(CreatedDate).minute
# len(f_year)


# In[235]:

g_year = pd.get_dummies(f_year)
g_quarter= pd.get_dummies(f_quarter)
g_month = pd.get_dummies(f_month)
g_weekno = pd.get_dummies(f_weekno)
g_dayofyr = pd.get_dummies(f_dayofyr)
g_weekday = pd.get_dummies(f_weekday)
g_day = pd.get_dummies(f_day)

# feature_full = pd.concat([g_year,g_quarter,g_month,g_weekday,g_day], axis=1)  
# feature_full = pd.concat([g_year,g_quarter,g_month,g_weekno,g_dayofyr,g_weekday,g_day], axis=1)  
feature_full = pd.concat([g_quarter,g_month,g_weekno,g_dayofyr,g_weekday,g_day], axis=1)  
feature_full.applymap(np.int) 
len(feature_full.keys())
len(feature_full)


# In[65]:

len(complaints_volume.index[complaints_volume.index<'2012-07-01'])


# In[67]:

len(complaints_volume.index[complaints_volume.index<'2013-07-01'])


# In[54]:

complaints_volume.index<2012


# In[236]:

X_train = feature_full[:1100]
X_test = feature_full[1100:1300]
Y_train = complaints_volume[:1100]
Y_test = complaints_volume[1100:1300]


# In[218]:

linregmodel = linear_model.LinearRegression()


# In[237]:

linregmodel.fit(X_train,Y_train)
Ytrainpredicted = linregmodel.predict(X_train)


# In[146]:

#for logisitc regression
# from sklearn import metrics
# metrics.accuracy_score(Y_test, Ytestpredicted)*100
# metrics.accuracy_score(Y_train, Ytrainpredicted)*100


# In[238]:

linregmodel.score(X_train,Y_train)


# In[239]:

RMSE_train = (((Ytrainpredicted-Y_train)**2).mean())**(0.5)
RMSE_train


# In[240]:

RMSE_train_rel = ((((Ytrainpredicted-Y_train)/Y_train)**2).mean())**(0.5)
RMSE_train_rel


# In[241]:

Ytestpredicted = linregmodel.predict(X_test)


# In[242]:

plot(complaints_volume.index[:1100],Y_train,'b',complaints_volume.index[:1100],Ytrainpredicted,'r')


# In[234]:




# In[244]:

plot(complaints_volume.index[1100:1300],Y_test,'b',complaints_volume.index[1100:1300],Ytestpredicted,'r')


# In[245]:

plot(complaints_volume.index,complaints_volume,'b',complaints_volume.index[1100:1300],Ytestpredicted,'r')


# In[158]:

complaints_volume.index[1000]


# In[159]:

complaints_volume.index[1300]


# In[ ]:



