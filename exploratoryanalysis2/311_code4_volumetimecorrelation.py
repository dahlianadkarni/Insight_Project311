
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
# from sklearn import preprocessing, linear_model
# import scipy.stats
# import scipy
# import math

mydb = mdb.connect(host='localhost',
    user='root',
    passwd='',
    db='311db_till2013')


# In[6]:

topagency = psql.read_sql("SELECT agency, count(*) FROM complaints GROUP BY agency ORDER BY COUNT(*) DESC", con=mydb)
topagency.head()


# In[116]:

# topcomplaintsHPD = psql.read_sql("SELECT complaint_type, count(*) FROM complaints WHERE agency='HPD'  GROUP BY complaint_type ORDER BY COUNT(*) DESC", con=mydb)
# topcomplaintsHPD.head()

topcomplaintsHPD = psql.read_sql("SELECT complaint_type, count(*) FROM complaints WHERE agency='DOT'  GROUP BY complaint_type ORDER BY COUNT(*) DESC", con=mydb)
topcomplaintsHPD.head()


# In[117]:

# topcomplaintsHPD = psql.read_sql("SELECT complaint_type, incident_zip, count(*) FROM complaints WHERE closed_date IS NOT NULL AND agency='HPD'  GROUP BY complaint_type,incident_zip ORDER BY COUNT(*) DESC", con=mydb)
# topcomplaintsHPD.head()

# topcomplaintsHPD = psql.read_sql("SELECT complaint_type, descriptor, count(*) FROM complaints WHERE closed_date IS NOT NULL AND agency='HPD'  GROUP BY complaint_type,descriptor ORDER BY COUNT(*) DESC", con=mydb)
# topcomplaintsHPD.head()


# In[105]:

# topcomplaintsHPD = psql.read_sql("SELECT incident_zip,complaint_type,  count(*) FROM complaints WHERE closed_date IS NOT NULL AND agency='HPD'  AND complaint_type!='HEATING' GROUP BY complaint_type,incident_zip ORDER BY COUNT(*) DESC", con=mydb)
# topcomplaintsHPD.head()
topcomplaintsHPD = psql.read_sql("SELECT incident_zip,complaint_type,descriptor, count(*) FROM complaints WHERE closed_date IS NOT NULL AND complaint_type='GENERAL CONSTRUCTION' AND incident_zip=11226 GROUP BY descriptor ORDER BY COUNT(*) DESC", con=mydb)
topcomplaintsHPD.head()


# In[142]:

# # # complaint_type='GENERAL CONSTRUCTION' AND incident_zip=11226
# dataclosed = psql.read_sql("SELECT created_date,closed_date,descriptor,due_date,status,latitude,longitude FROM complaints WHERE closed_date IS NOT NULL AND created_date IS NOT NULL AND complaint_type='GENERAL CONSTRUCTION' AND incident_zip=11226", parse_dates=['created_date','closed_date'], con=mydb)

# # # descriptor='MOLD'
# dataclosed = psql.read_sql("SELECT created_date, closed_date, agency, incident_zip, complaint_type, descriptor, due_date FROM complaints WHERE closed_date IS NOT NULL AND created_date IS NOT NULL AND descriptor='MOLD'", parse_dates=['created_date','closed_date'], con=mydb)

# # # descriptor='Street%'
# dataclosed = psql.read_sql("SELECT created_date, closed_date, agency, incident_zip, complaint_type, descriptor, due_date FROM complaints WHERE closed_date IS NOT NULL AND created_date IS NOT NULL AND complaint_type LIKE 'Street%'", parse_dates=['created_date','closed_date'], con=mydb)

# # # complaint_type='Street%'
# dataclosed = psql.read_sql("SELECT created_date, closed_date, agency, incident_zip, complaint_type, descriptor, due_date FROM complaints WHERE closed_date IS NOT NULL AND created_date IS NOT NULL AND complaint_type='Street Light Condition'", parse_dates=['created_date','closed_date'], con=mydb)

# # complaint_type='Rodent%'
dataclosed = psql.read_sql("SELECT created_date, closed_date, agency, incident_zip, complaint_type, descriptor, due_date FROM complaints WHERE closed_date IS NOT NULL AND created_date IS NOT NULL AND complaint_type='Rodent'", parse_dates=['created_date','closed_date'], con=mydb)

dataclosed.head()


# In[143]:

set(dataclosed['descriptor'])
# dataclosed = dataclosed[dataclosed['descriptor']=='HEAT']


# In[144]:

# complaints = dataclosed[['created_date','descriptor']]
index = dataclosed['created_date']>'2010-01-01'
dataclosed = dataclosed[index]
complaints = dataclosed[['created_date','descriptor']]
complaints_volume= complaints.set_index('created_date').sort_index().resample('W', how=len)
# complaints_volume
complaints_volume.plot()
len(complaints_volume)
# complaints_volume.plot()


# In[150]:

ClosedDate = pd.to_datetime(pd.Series(dataclosed['closed_date']))
CreatedDate = pd.to_datetime(pd.Series(dataclosed['created_date']))
ActualTime = ClosedDate-CreatedDate
ActualTime = ActualTime.astype('timedelta64[D]')
index1 = (ActualTime>0)
ActualTime = ActualTime[index1]
dataclosed1 = dataclosed[index1]
dataclosed1['actual_time']=ActualTime

ResTimesAct = dataclosed1[['created_date','actual_time']].set_index('created_date').sort_index().resample('W', how='mean')
ResTimesAct.plot()
len(ResTimesAct)

complaints = dataclosed[['created_date','descriptor']][index1]
complaints_volume= complaints.set_index('created_date').sort_index().resample('W', how=len)
complaints_volume.plot()
len(complaints_volume)

# ResTimesAct['ResTimeActual']
# ResTimesAct = pd.DataFrame(ResTimesAct)
# ResTimesAct = ResTimesAct.set_index('Created Date').sort_index()
# ResTimesAct
# # ResTimesAct.astype('timedelta64[m]').mean()
# ResTimesAct.astype('timedelta64[m]').std()
# ResTimesAct.astype('timedelta64[m]').resample('D', how='mean').plot()


# In[152]:

# plot(ResTimesAct,log(complaints_volume),'.')
# plot(log(ResTimesAct),log(complaints_volume),'.')
plot(ResTimesAct,complaints_volume,'.')


# In[ ]:



