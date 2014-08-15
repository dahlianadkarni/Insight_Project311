###########

# from app import app
# 
# @app.route('/')
# @app.route('/index')
# def index():
#     return "Hello, World!"

###########

# from app import app
# 
# @app.route('/')
# @app.route('/index')
# def index():
#     user = { 'nickname': 'Miguel' } # fake user
#     return '''
# <html>
#   <head>
#     <title>Home Page</title>
#   </head>
#   <body>
#     <h1>Hello, ''' + user['nickname'] + '''</h1>
#   </body>
# </html>
# '''

###########

# from flask import render_template
# from app import app
# 
# @app.route('/')
# @app.route('/index')
# def index():
#     user = { 'nickname': 'Miguel' } # fake user
#     return render_template("index1.html",
#         title = 'Home',
#         user = user)

###########


from flask import render_template, jsonify, Flask
from app import app
import pymysql as mdb

db = mdb.connect(user="root", host="localhost", db="311db_till2013", charset='utf8')

@app.route('/')
@app.route('/index')
def index():
	return render_template("index.html",
        title = 'Home', user = { 'nickname': 'Miguel' },
        )

@app.route('/db')
def complaints_page():
	with db: 
		cur = db.cursor()
		cur.execute("SELECT complaint_type FROM complaints LIMIT 15;")
		query_results = cur.fetchall()
	complaints = ""
	for result in query_results:
		complaints += result[0]
		complaints += "<br>"
	return complaints
	
@app.route("/db_fancy")
def complaints_page_fancy():
	with db:
		cur = db.cursor()
		cur.execute("SELECT created_date,complaint_type,borough FROM complaints LIMIT 15;")

		query_results = cur.fetchall()
	complaints = []
	for result in query_results:
		complaints.append(dict(created_date=result[0], complaint_type=result[1], borough=result[2]))
	return render_template('complaints.html', complaints=complaints) 
	
@app.route("/db_json")
def complaints_json():
    with db:
        cur = db.cursor()
#         cur.execute("SELECT created_date,complaint_type,borough FROM complaints ORDER BY created_date DESC LIMIT 15;")
        cur.execute("SELECT created_date,complaint_type,borough FROM complaints LIMIT 15;")
        query_results = cur.fetchall()
    complaints = []
    for result in query_results:
        complaints.append(dict(created_date=result[0], complaint_type=result[1], borough=result[2]))
    return jsonify(dict(complaints=complaints))	
    
# @app.route("/jquery")
# def index_jquery():
# 	return render_template('index_js_my.html')    
	
@app.route("/jquery2")
def index_jquery2():
	return render_template('index_js_my_new.html')    	
	
@app.route("/rank")

def cities_rank():
        print request.args.get('keywords', '')
        user_input = request.args.get('keywords', '')
	user_input = user_input.lstrip().rstrip()

        country_input = request.args.get('country', '')
	country_input = country_input.lstrip().rstrip()

	with db:
		cur = db.cursor()
		cmnd = "SELECT Name,Population FROM City Limit 15 "

#		if len(user_input.split()) > 0:
#			cmnd = cmnd + "WHERE ("
			# SELECT title, top_words FROM ranking WHERE 
			# (search_terms LIKE "%pub%" AND search_terms LIKE "%bar%") AND
			# (region LIKE "% UNITED %" OR region LIKE "% Australia %");
#			for st in user_input.split():
#				cmnd = cmnd + " search_terms LIKE '%"
#				cmnd = cmnd + "%s" % st
#				cmnd = cmnd + "%'"
#				if user_input.split()[-1] != st:
#					cmnd = cmnd + " AND "
#			cmnd = cmnd + ")"
#
#		if country_input != '':
#			if len(user_input.split()) > 0:
#				cmnd = cmnd + ' AND '
#			else:
#				cmnd = cmnd + 'WHERE '
#			cmnd = cmnd + '('
#      	        	for st in country_input.split(','):
#                      		cmnd = cmnd + " region LIKE '% "
#                       	cmnd = cmnd + "%s" % st
#                        	cmnd = cmnd + " %'"
#                        	if country_input.split(',')[-1] != st:
#                                	cmnd = cmnd + " OR "
#
#			cmnd = cmnd + ");"

		print cmnd
		cur.execute(cmnd)
		query_results = cur.fetchall()
	cities = []
	for result in query_results:
		cities.append(dict(title=result[0], search_terms=result[1]))

	#return "<h3>This is the server response!</h3>"
	tmp = '<table id = "ranklist" class="table table-hover">'
    	tmp = tmp + '<tr><th>Name</th><th>Key Words</th></tr>'

	for city in cities:
    		tmp = tmp + '<tr><td>' + str(city["title"]) + '</td><td>'+ str(city["search_terms"]) + '</td></tr>'
    		
    		
	#tmp = tmp + '<tr><td>' + city["Name"] + '</td><td>'+ city["Population"] + '</td></tr>'    		
	
	tmp = tmp + '</table>'

	return tmp