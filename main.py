from flask import Flask, jsonify, request
import json

app = Flask(__name__)

@app.route("/", methods = ["GET", "POST"])
def index():
    if(request.method == "GET"):
        return jsonify({"data": "TESTING THIS API"})

    if(request.method == "POST"):
        data = json.loads(request.data.decode("utf-8"))
        print(data)

        return data

@app.route("/speech", methods = ["POST"])
def speech():
    if(request.method == "POST"):
        data = json.loads(request.data.decode("utf-8"))
        print(data)

        return data["text"]

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=9999)
