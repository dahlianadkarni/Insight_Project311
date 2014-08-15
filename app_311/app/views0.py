from flask import render_template, Flask, jsonify
from app import app
import pymysql as mdb

db = mdb.connect(user="root", host="localhost", db="311db_till2013", charset='utf8')

@app.route('/')
@app.route('/jquery')
def index_jquery():
    return render_template('index_js.html')

@app.route('/db')
def cities_page():
	with db: 
		cur = db.cursor()
# 		cur.execute("SELECT complaint_type FROM complaints LIMIT 15;")
		cur.execute("SELECT complaint_type, count(unique_key) FROM complaints GROUP BY complaint_type ORDER BY count(unique_key) DESC LIMIT 15;")
		query_results = cur.fetchall()
	cities = ""
	for result in query_results:
		cities += result[0]
# 		cities += result[1]
		cities += "<br>"
	return cities
	

@app.route("/db_fancy")
def cities_page_fancy():
	with db:
		cur = db.cursor()
# 		cur.execute("SELECT complaint_type, count(*) FROM complaints GROUP BY complaint_type ORDER BY count(*) DESC LIMIT 15;")
		cur.execute("SELECT complaint_type FROM complaints GROUP BY complaint_type ORDER BY count(*) DESC;")
		query_results = cur.fetchall()
	cities = []
	for result in query_results:
	    cities.append(dict(complaint_type=result[0]))
# 		cities.append(dict(complaints=result[0], counts=result[1]))
	return render_template('complaints.html', cities=cities) 

@app.route("/db_json")
def cities_json():
    with db:
        cur = db.cursor()
#         cur.execute("SELECT Name, CountryCode, Population FROM city ORDER BY Population DESC;")
        cur.execute("SELECT complaint_type, count(*) FROM complaints GROUP BY complaint_type ORDER BY count(*) DESC LIMIT 15;")
        query_results = cur.fetchall()
    cities = []
    for result in query_results:
#         cities.append(dict(name=result[0], country=result[1], population=result[2]))
        cities.append(dict(complaint_type=result[0], count = result[1]))
    return jsonify(dict(cities=cities))
