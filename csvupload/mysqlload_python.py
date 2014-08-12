import csv
import pymysql as mdb
import numpy as np
import time
import sys
import re
from datetime import datetime

startTime = datetime.now()

mydb_alltext = mdb.connect(host='localhost',
    user='root',
    passwd='',
    db='311db_alltext')
cursor = mydb_alltext.cursorsor()

# drop table - complaint - and create them new
cursor.execute('DROP TABLE IF EXISTS complaints')


input_file = '../311Data/311_Service_Requests_from_2010_to_Present_2.csv'
nrows = 7790056
ncols = 52
# input_file = '311_Service_Requests_from_2010_to_Present.csv'
# # nrows = 2122265
# # ncols = 52
# 
# 
# # get size of table in not known apriori
# mytable_readfromcsv = csv.reader(file(input_file))
# flag = False
# header = mytable_readfromcsv.next()
# nrows = 0
# for row in mytable_readfromcsv:
#     nrows = nrows+1
# print nrows    
# ncols = len(row)
# print ncols
# 
# print(datetime.now()-startTime)
# 
# # 
# 
# # # get maximum width of columns in table complaints
# mytable_readfromcsv = csv.reader(file(input_file))
# header = mytable_readfromcsv.next()
# maxwidth = np.zeros((ncols))
# countdatashifted=0
# for i,row in enumerate(mytable_readfromcsv):
#     for ind in indexes_sort:
#         del row[ind]
#     if i%100000 ==0:
#         print i
#         print(datetime.now()-startTime)
#     if len(row)==ncols:
#         for j,cell in enumerate(row):
#             if len(cell) > maxwidth[j]:
#                 maxwidth[j] = len(cell)
#     else:
#         countdatashifted += 1
# 
# print countdatashifted
# # # maxwidth = L.max(0)
# print maxwidth
# print(datetime.now()-startTime)

maxwidth =     [   8.   22.   22.    6.   91.   41.  106.   36.   10.   81.   80.   36.
   36.   38.   47.   12.   32.   48.   15.   28.   22.   22.   25.   13.
    7.    7.   95.   13.   95.    8.   27.    6.   10.  120.   19.    2.
    5.    2.   18.   23.   13.   27.   42.   30.    7.  100.   27.   19.
   95.   18.   18.   40.]
# # read in data and put into table
mytable_readfromcsv = csv.reader(file(input_file))
header = mytable_readfromcsv.next()
headersql = [s.replace('-','_').replace(' ','_').replace('\n','_').replace('*','').replace('(','').replace(')','') for s in header]

querylist = []
for i in range(ncols):
    querylist.append(headersql[i] + ' VARCHAR(' + str(int(maxwidth[i])) + ')')

query = 'CREATE TABLE complaints(' + ",".join(querylist) + ')'
cursor.execute(query)

query = 'INSERT INTO complaints(' + '%s, ' *(len(headersql)-1) + '%s)'
query = query % tuple(headersql)
query = query + ' VALUES(' + '%s, ' *(len(headersql)-1) + '%s)'

print(datetime.now()-startTime)

i = 0
for row in mytable_readfromcsv:
    if len(row)==ncols:
        cursor.execute(query, row)
        i+=1
        if i%100000==0:
            print i
            print(datetime.now()-startTime)
print i

mydb_alltext.commit()

#close mysql cursorsor
cursor.close()
print "Done"

print(datetime.now()-startTime)