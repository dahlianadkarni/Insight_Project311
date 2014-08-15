from flask import Flask,render_template,jsonify,request
from app import app
from cStringIO import StringIO
from pandas.io import sql

import numpy as np
import pymysql as psql
import pandas as pd
import matplotlib.pyplot as plt

conn = psql.connect(user="root", passwd="", host="localhost",
                    db="ny_hospitals", charset='utf8')

def age_to_age_group(age):
    age_group = '0 to 17'
    if   age >= 18 and age < 30:
        age_group = '18 to 29'
    elif age >= 30 and age < 49:
        age_group = '30 to 49'
    elif age >= 49 and age < 69:
        age_group = '50 to 69'
    else:
        age_group = '70 or Older'

    return age_group

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

def print_result(df):
    plt.subplots(1, 1, sharex=True, figsize=(8, 6))
    for y in set(df.discharge_year):
        ax = np.log(df.total_charges[df.discharge_year==y]).plot(kind='kde',
                                                                 legend=True,
                                                                 label=y)
    ax.set_xlabel('Log-Total Charges')

    io = StringIO()
    plt.savefig(io, format='png', dpi=150)
    return io.getvalue().encode('base64')

@app.route('/echo/', methods=['GET'])
def echo():
    ret_data = {'diagValue': request.args.get('diagValue'),
                'ageValue':  request.args.get('ageValue')}

    age_group = age_to_age_group(int(ret_data['ageValue']))

    with conn:
        #curu= conn.cursor()
        #query = "SELECT facility_name, AVG(total_charges)                     \
        #         FROM hospital_discharge                                      \
        #         WHERE ccs_procedure_code = '%" + ret_data['diagValue'] + "%' \
        #         AND                                                          \
        #         age_group     = '" + age_group +                   "'        \
        #         GROUP BY facility_name                                       \
        #         ORDER BY AVG(total_charges)"
        #cur.execute(query)
        #query_results = cur.fetchall()
        query = "SELECT total_charges, discharge_year \
                 FROM hospital_discharge \
                 WHERE ccs_procedure_code=" + ret_data['diagValue'] + " AND \
                       age_group='" + age_group + "'"
        df = sql.read_sql(query, conn)

    hospital = []
    for i, result in enumerate(df.to_records(index=False)):
        if i > 10:
            break
        hospital.append(dict(name=result[0],
                             avr_total_charges=result[1]))

    return jsonify(dict(hospital=hospital, image=print_result(df)))