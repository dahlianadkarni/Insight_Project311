
# coding: utf-8

# In[60]:

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
pd.options.display.mpl_style = 'default'
import datetime

from sklearn import preprocessing



# In[12]:

# data1007 = pd.read_csv('/Users/dahlia/Documents/INSIGHT PROJECT/Data/311/311_Service_Requests_2010_07_NYPDnoise_zip1000.csv', parse_dates=['Created Date','Closed Date'],low_memory=False)
# data1107 = pd.read_csv('/Users/dahlia/Documents/INSIGHT PROJECT/Data/311/311_Service_Requests_2011_07_NYPDnoise_zip1000.csv', parse_dates=['Created Date','Closed Date'],low_memory=False)
# data1207 = pd.read_csv('/Users/dahlia/Documents/INSIGHT PROJECT/Data/311/311_Service_Requests_2012_07_NYPDnoise_zip1000.csv', parse_dates=['Created Date','Closed Date'],low_memory=False)
# data1307 = pd.read_csv('/Users/dahlia/Documents/INSIGHT PROJECT/Data/311/311_Service_Requests_2013_07_NYPDnoise_zip1000.csv', parse_dates=['Created Date','Closed Date'],low_memory=False)
# dataclosed1007 = data1007[[u'Unique Key', u'Created Date', u'Closed Date', u'Agency', u'Agency Name', u'Complaint Type', u'Descriptor', u'Incident Zip', u'City', u'Status', u'Due Date', u'Resolution Action Updated Date', u'Community Board', u'Borough', u'X Coordinate (State Plane)', u'Y Coordinate (State Plane)', u'Latitude', u'Longitude', u'Location']].dropna()
# dataclosed1107 = data1107[[u'Unique Key', u'Created Date', u'Closed Date', u'Agency', u'Agency Name', u'Complaint Type', u'Descriptor', u'Incident Zip', u'City', u'Status', u'Due Date', u'Resolution Action Updated Date', u'Community Board', u'Borough', u'X Coordinate (State Plane)', u'Y Coordinate (State Plane)', u'Latitude', u'Longitude', u'Location']].dropna()
# dataclosed1207 = data1207[[u'Unique Key', u'Created Date', u'Closed Date', u'Agency', u'Agency Name', u'Complaint Type', u'Descriptor', u'Incident Zip', u'City', u'Status', u'Due Date', u'Resolution Action Updated Date', u'Community Board', u'Borough', u'X Coordinate (State Plane)', u'Y Coordinate (State Plane)', u'Latitude', u'Longitude', u'Location']].dropna()
# dataclosed1307 = data1307[[u'Unique Key', u'Created Date', u'Closed Date', u'Agency', u'Agency Name', u'Complaint Type', u'Descriptor', u'Incident Zip', u'City', u'Status', u'Due Date', u'Resolution Action Updated Date', u'Community Board', u'Borough', u'X Coordinate (State Plane)', u'Y Coordinate (State Plane)', u'Latitude', u'Longitude', u'Location']].dropna()

data10 = pd.read_csv('/Users/dahlia/Documents/INSIGHT PROJECT/Data/311/311_Service_Requests_2010_NYPDnoise_zip1000.csv', parse_dates=['Created Date','Closed Date'],low_memory=False)
data11 = pd.read_csv('/Users/dahlia/Documents/INSIGHT PROJECT/Data/311/311_Service_Requests_2011_NYPDnoise_zip1000.csv', parse_dates=['Created Date','Closed Date'],low_memory=False)
data12 = pd.read_csv('/Users/dahlia/Documents/INSIGHT PROJECT/Data/311/311_Service_Requests_2012_NYPDnoise_zip1000.csv', parse_dates=['Created Date','Closed Date'],low_memory=False)
data13 = pd.read_csv('/Users/dahlia/Documents/INSIGHT PROJECT/Data/311/311_Service_Requests_2013_NYPDnoise_zip1000.csv', parse_dates=['Created Date','Closed Date'],low_memory=False)
dataclosed10 = data10[[u'Unique Key', u'Created Date', u'Closed Date', u'Agency', u'Agency Name', u'Complaint Type', u'Descriptor', u'Incident Zip', u'City', u'Status', u'Due Date', u'Resolution Action Updated Date', u'Community Board', u'Borough', u'X Coordinate (State Plane)', u'Y Coordinate (State Plane)', u'Latitude', u'Longitude', u'Location']].dropna()
dataclosed11 = data11[[u'Unique Key', u'Created Date', u'Closed Date', u'Agency', u'Agency Name', u'Complaint Type', u'Descriptor', u'Incident Zip', u'City', u'Status', u'Due Date', u'Resolution Action Updated Date', u'Community Board', u'Borough', u'X Coordinate (State Plane)', u'Y Coordinate (State Plane)', u'Latitude', u'Longitude', u'Location']].dropna()
dataclosed12 = data12[[u'Unique Key', u'Created Date', u'Closed Date', u'Agency', u'Agency Name', u'Complaint Type', u'Descriptor', u'Incident Zip', u'City', u'Status', u'Due Date', u'Resolution Action Updated Date', u'Community Board', u'Borough', u'X Coordinate (State Plane)', u'Y Coordinate (State Plane)', u'Latitude', u'Longitude', u'Location']].dropna()
dataclosed13 = data13[[u'Unique Key', u'Created Date', u'Closed Date', u'Agency', u'Agency Name', u'Complaint Type', u'Descriptor', u'Incident Zip', u'City', u'Status', u'Due Date', u'Resolution Action Updated Date', u'Community Board', u'Borough', u'X Coordinate (State Plane)', u'Y Coordinate (State Plane)', u'Latitude', u'Longitude', u'Location']].dropna()



# In[62]:

vec=dataclosed1007['Created Date'][:10]
# vec = pd.to_datetime(vec)
datetime.utcfromtimestamp(vec)
vec


# In[7]:

# data = pd.read_csv('/Users/dahlia/Documents/INSIGHT PROJECT/Data/311/311_Service_Requests_2010to13_NYPDnoise_zip1000.csv', parse_dates=['Created Date','Closed Date','Due Date'],low_memory=False)
# dataclosed = data[[u'Unique Key', u'Created Date', u'Closed Date', u'Agency', u'Agency Name', u'Complaint Type', u'Descriptor', u'Incident Zip', u'City', u'Status', u'Due Date', u'Resolution Action Updated Date', u'Community Board', u'Borough', u'X Coordinate (State Plane)', u'Y Coordinate (State Plane)', u'Latitude', u'Longitude', u'Location']].dropna()
# dataclosed[:][dataclosed['Actual Time']<=0]
# len(data)


# In[36]:

len(dataclosed['Expected Time'][dataclosed['Expected Time']<=0])


# In[48]:

data = pd.read_csv('/Users/dahlia/Documents/INSIGHT PROJECT/Data/311/311_Service_Requests_2010to13_NYPDnoise_zip1000.csv', parse_dates=['Created Date','Closed Date'],low_memory=False)
dataclosed = data[[u'Unique Key', u'Created Date', u'Closed Date', u'Agency', u'Agency Name', u'Complaint Type', u'Descriptor', u'Incident Zip', u'City', u'Status', u'Due Date', u'Resolution Action Updated Date', u'Community Board', u'Borough', u'X Coordinate (State Plane)', u'Y Coordinate (State Plane)', u'Latitude', u'Longitude', u'Location']].dropna()



# In[7]:

datanow = pd.read_csv('/Users/dahlia/Documents/INSIGHT PROJECT/Data/311/311_Service_Requests_2010to14_NYPDnoise_zip1000.csv', parse_dates=['Created Date','Closed Date'],low_memory=False)
dataclosednow = datanow[[u'Unique Key', u'Created Date', u'Closed Date', u'Agency', u'Agency Name', u'Complaint Type', u'Descriptor', u'Incident Zip', u'City', u'Status', u'Due Date', u'Resolution Action Updated Date', u'Community Board', u'Borough', u'X Coordinate (State Plane)', u'Y Coordinate (State Plane)', u'Latitude', u'Longitude', u'Location']].dropna()


# In[28]:




# In[14]:

ComplaintTypeCount = dataclosednow['Complaint Type'].value_counts()
ComplaintTypefrac = ComplaintTypeCount/sum(ComplaintTypeCount)
ComplaintTypeCount.plot(kind='bar')
ComplaintTypefrac


# In[66]:

ComplaintTypeCount = dataclosed1007['Complaint Type'].value_counts()

ComplaintTypefrac = ComplaintTypeCount/sum(ComplaintTypeCount)
# ComplaintTypeCount.plot(kind='bar')
ComplaintTypefrac


# In[67]:

ComplaintTypeCount = dataclosed1107['Complaint Type'].value_counts()
ComplaintTypefrac = ComplaintTypeCount/sum(ComplaintTypeCount)
# ComplaintTypeCount.plot(kind='bar')
ComplaintTypefrac


# In[40]:

ComplaintTypeCount = dataclosed1207['Complaint Type'].value_counts()
ComplaintTypefrac = ComplaintTypeCount/sum(ComplaintTypeCount)
# ComplaintTypeCount.plot(kind='bar')
ComplaintTypefrac


# In[8]:

ComplaintTypeCount = dataclosed1307['Complaint Type'].value_counts()
ComplaintTypefrac = ComplaintTypeCount/sum(ComplaintTypeCount)
# ComplaintTypeCount.plot(kind='bar')
ComplaintTypefrac


## complaints = dataclosed1007[['Created Date', 'Complaint Type']] complaints.set_index('Created Date').sort_index().resample('H', how=len).plot() complaints = dataclosed1107[['Created Date', 'Complaint Type']] complaints.set_index('Created Date').sort_index().resample('H', how=len).plot() complaints = dataclosed1207[['Created Date', 'Complaint Type']] complaints.set_index('Created Date').sort_index().resample('H', how=len).plot() complaints = dataclosed1307[['Created Date', 'Complaint Type']] complaints.set_index('Created Date').sort_index().resample('H', how=len).plot()

# In[6]:

complaints = dataclosed10[['Created Date', 'Complaint Type']]
noise_complaints = complaints[complaints['Complaint Type'] == 'Noise - Street/Sidewalk']
noise_complaints.set_index('Created Date').sort_index().resample('D', how=len).plot()
complaints = dataclosed11[['Created Date', 'Complaint Type']]
noise_complaints = complaints[complaints['Complaint Type'] == 'Noise - Street/Sidewalk']
noise_complaints.set_index('Created Date').sort_index().resample('D', how=len).plot()
complaints = dataclosed12[['Created Date', 'Complaint Type']]
noise_complaints = complaints[complaints['Complaint Type'] == 'Noise - Street/Sidewalk']
noise_complaints.set_index('Created Date').sort_index().resample('D', how=len).plot()
complaints = dataclosed13[['Created Date', 'Complaint Type']]
noise_complaints = complaints[complaints['Complaint Type'] == 'Noise - Street/Sidewalk']
noise_complaints.set_index('Created Date').sort_index().resample('D', how=len).plot()


# In[47]:

complaints = dataclosed10[['Created Date', 'Complaint Type']]
noise_complaints = complaints[complaints['Complaint Type'] == 'Noise - Street/Sidewalk']
noise_complaints.set_index('Created Date').sort_index().resample('D', how=len).plot()
complaints = dataclosed11[['Created Date', 'Complaint Type']]
noise_complaints = complaints[complaints['Complaint Type'] == 'Noise - Street/Sidewalk']
noise_complaints.set_index('Created Date').sort_index().resample('D', how=len).plot()
complaints = dataclosed12[['Created Date', 'Complaint Type']]
noise_complaints = complaints[complaints['Complaint Type'] == 'Noise - Street/Sidewalk']
noise_complaints.set_index('Created Date').sort_index().resample('D', how=len).plot()
complaints = dataclosed13[['Created Date', 'Complaint Type']]
noise_complaints = complaints[complaints['Complaint Type'] == 'Noise - Street/Sidewalk']
noise_complaints.set_index('Created Date').sort_index().resample('D', how=len).plot()


# In[52]:

complaints = dataclosed[['Created Date', 'Complaint Type']]
noise_complaints = complaints[complaints['Complaint Type'] == 'Noise - Street/Sidewalk']
noise_complaints.set_index('Created Date').sort_index().resample('W', how=len).plot()


# In[53]:

complaints = dataclosed[['Created Date', 'Complaint Type']]
noise_complaints = complaints[complaints['Complaint Type'] == 'Noise - Street/Sidewalk']
noise_complaints.set_index('Created Date').sort_index().resample('D', how=len).plot()


# In[8]:

complaints = dataclosednow[['Created Date', 'Complaint Type']]
noise_complaints = complaints[complaints['Complaint Type'] == 'Noise - Street/Sidewalk']
noise_complaints.set_index('Created Date').sort_index().resample('W', how=len).plot()


# In[9]:

complaints = dataclosednow[['Created Date', 'Complaint Type']]
noise_complaints = complaints[complaints['Complaint Type'] == 'Noise - Street/Sidewalk']
noise_complaints.set_index('Created Date').sort_index().resample('D', how=len).plot()


# In[12]:

complaints = dataclosednow[['Created Date', 'Complaint Type']]
noise_complaints = complaints[complaints['Complaint Type'] == 'Noise - Street/Sidewalk']
noise_complaints.set_index('Created Date').sort_index().resample('M', how=len).plot()


# In[77]:

data1 = dataclosed10[dataclosed10['Incident Zip'] == 10001]
ComplaintTypeCount = data1['Complaint Type'].value_counts()
ComplaintTypefrac = ComplaintTypeCount/sum(ComplaintTypeCount)
# ComplaintTypeCount.plot(kind='bar')
ComplaintTypefrac


# In[101]:

data2 = dataclosed11[dataclosed11['Incident Zip'] == 10005]
ComplaintTypeCount = data2['Complaint Type'].value_counts()
ComplaintTypefrac5 = ComplaintTypeCount/sum(ComplaintTypeCount)
ComplaintTypeCount.plot(kind='bar')
ComplaintTypefrac5


# In[100]:

data2 = dataclosed11[dataclosed11['Incident Zip'] == 10000]
ComplaintTypeCount = data2['Complaint Type'].value_counts()
ComplaintTypefrac0 = ComplaintTypeCount/sum(ComplaintTypeCount)
ComplaintTypeCount.plot(kind='bar')
ComplaintTypefrac0


# In[102]:

data2 = dataclosed11[dataclosed11['Incident Zip'] == 10009]
ComplaintTypeCount = data2['Complaint Type'].value_counts()
ComplaintTypefrac9 = ComplaintTypeCount/sum(ComplaintTypeCount)
ComplaintTypeCount.plot(kind='bar')
ComplaintTypefrac9


# In[152]:

complaints = dataclosed1207[['Created Date', 'Complaint Type']]
noise_complaints = complaints[complaints['Complaint Type'] == 'Noise - Street/Sidewalk']
noise_complaints.set_index('Created Date').sort_index().resample('D', how=len).plot()
vec1 = noise_complaints.set_index('Created Date').sort_index().resample('D', how=len)


# In[151]:

complaints = dataclosed1107[['Created Date', 'Complaint Type']]
noise_complaints = complaints[complaints['Complaint Type'] == 'Noise - Street/Sidewalk']
noise_complaints.set_index('Created Date').sort_index().resample('D', how=len).plot()
vec2 = noise_complaints.set_index('Created Date').sort_index().resample('D', how=len)


# In[145]:

vec1


# In[132]:

vec2


# In[28]:




# In[28]:

index1 = dataclosed['Actual Time']>0 
index2 = dataclosed['Expected Time']>0
index = index1 & index2
dataclosed = dataclosed[:][index]
len(dataclosed)


# In[153]:

StatusCounts = data['Status'].value_counts()
StatusCounts.plot(kind='bar')
StatusCounts


# In[154]:

data1007


# In[157]:

ActualTimeDiff = data1007['Closed Date']-data1007['Created Date']
ActualTimeDiffNoise = ActualTimeDiff[:][data1007['Complaint Type']=='Noise - Street/Sidewalk']
ActualTimeDiffNoise.mean()



# In[211]:





# In[210]:




# In[211]:




# In[211]:

data100 = pd.read_csv('/Users/dahlia/Documents/INSIGHT PROJECT/Data/311/311_Service_Requests_2010_07_NYPDandDOHMH_zip1000.csv', parse_dates=['Created Date','Closed Date'],low_memory=False)
dataclosed100 = data100[[u'Unique Key', u'Created Date', u'Closed Date', u'Agency', u'Agency Name', u'Complaint Type', u'Descriptor', u'Incident Zip', u'City', u'Status', u'Due Date', u'Resolution Action Updated Date', u'Community Board', u'Borough', u'X Coordinate (State Plane)', u'Y Coordinate (State Plane)', u'Latitude', u'Longitude', u'Location']].dropna()
complaints = dataclosed100[['Created Date', 'Complaint Type']]


# In[220]:


noise_complaints = complaints[complaints['Complaint Type'] == 'Rodent']
noise_complaints.set_index('Created Date').sort_index().resample('D', how=len).plot()
vec2 = noise_complaints.set_index('Created Date').sort_index().resample('D', how=len)


# In[213]:

data100 = pd.read_csv('/Users/dahlia/Documents/INSIGHT PROJECT/Data/311/311_Service_Requests_2011_07_NYPDandDOHMH_zip1000.csv', parse_dates=['Created Date','Closed Date'],low_memory=False)
dataclosed100 = data100[[u'Unique Key', u'Created Date', u'Closed Date', u'Agency', u'Agency Name', u'Complaint Type', u'Descriptor', u'Incident Zip', u'City', u'Status', u'Due Date', u'Resolution Action Updated Date', u'Community Board', u'Borough', u'X Coordinate (State Plane)', u'Y Coordinate (State Plane)', u'Latitude', u'Longitude', u'Location']].dropna()
complaints = dataclosed100[['Created Date', 'Complaint Type']]
noise_complaints = complaints[complaints['Complaint Type'] == 'Rodent']
noise_complaints.set_index('Created Date').sort_index().resample('D', how=len).plot()
vec2 = noise_complaints.set_index('Created Date').sort_index().resample('D', how=len)


# In[214]:

data100 = pd.read_csv('/Users/dahlia/Documents/INSIGHT PROJECT/Data/311/311_Service_Requests_2012_07_NYPDandDOHMH_zip1000.csv', parse_dates=['Created Date','Closed Date'],low_memory=False)
dataclosed100 = data100[[u'Unique Key', u'Created Date', u'Closed Date', u'Agency', u'Agency Name', u'Complaint Type', u'Descriptor', u'Incident Zip', u'City', u'Status', u'Due Date', u'Resolution Action Updated Date', u'Community Board', u'Borough', u'X Coordinate (State Plane)', u'Y Coordinate (State Plane)', u'Latitude', u'Longitude', u'Location']].dropna()
complaints = dataclosed100[['Created Date', 'Complaint Type']]
noise_complaints = complaints[complaints['Complaint Type'] == 'Rodent']
noise_complaints.set_index('Created Date').sort_index().resample('D', how=len).plot()
vec2 = noise_complaints.set_index('Created Date').sort_index().resample('D', how=len)


# In[215]:

data100 = pd.read_csv('/Users/dahlia/Documents/INSIGHT PROJECT/Data/311/311_Service_Requests_2013_07_NYPDandDOHMH_zip1000.csv', parse_dates=['Created Date','Closed Date'],low_memory=False)
dataclosed100 = data100[[u'Unique Key', u'Created Date', u'Closed Date', u'Agency', u'Agency Name', u'Complaint Type', u'Descriptor', u'Incident Zip', u'City', u'Status', u'Due Date', u'Resolution Action Updated Date', u'Community Board', u'Borough', u'X Coordinate (State Plane)', u'Y Coordinate (State Plane)', u'Latitude', u'Longitude', u'Location']].dropna()
complaints = dataclosed100[['Created Date', 'Complaint Type']]
noise_complaints = complaints[complaints['Complaint Type'] == 'Rodent']
noise_complaints.set_index('Created Date').sort_index().resample('D', how=len).plot()
vec2 = noise_complaints.set_index('Created Date').sort_index().resample('D', how=len)


# In[211]:




# In[208]:

ActualTimeDiff = data100['Closed Date']-data100['Created Date']
ActualTimeDiff = ActualTimeDiff[ActualTimeDiff>0]
ActualTimeDiffNoise = ActualTimeDiff[:][data100['Complaint Type']=='Rodent']
ActualTimeDiffNoise.mean()


# In[ ]:



