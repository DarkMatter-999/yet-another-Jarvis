from flask import Flask, jsonify, request
import json

# Services
from microservices.chatbot.chat import chat
from microservices.google_scraping.google import google_search
from microservices.weather.weather import weather
from microservices.joke.jokes import getjoke 

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
        query = data["text"]
        result = ""

        if query == "":
            return ""
        elif "google" in query:
            result = google_search(query.replace("google", "").replace("search", ""))
        elif "weather" in query:
            result = weather()
        elif "joke" in query:
            result = getjoke()
        else:
            result = chat(query)

        return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=9999)
