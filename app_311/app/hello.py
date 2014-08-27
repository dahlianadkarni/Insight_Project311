from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello():	
     print "hello" 
     return "Hello!"

#if __name__ == "__main":
#	app.run(host='0.0.0.0',port=5000)
if __name__ == "__main__":
     app.run(host='0.0.0.0', port = 5000) 
#    app.run(host='0.0.0.0') 
