import requests, json
from bs4 import BeautifulSoup

def weather():
    search = "whats the weather outside"
    search = search.replace(" ", "+")

    url = f"https://www.google.com/search?&q={search}"

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.99 Safari/537.36"}
    req = requests.get(url, headers=headers)
    sor = BeautifulSoup(req.text, "html.parser")

    try:
        temperature = sor.find("span", class_='wob_t q8U8x').text
        location = sor.find("div", class_='wob_loc q8U8x').text
        weather_ = sor.find("div", class_='wob_dcp').text

        temp = "it's" + temperature + " degree celcius in " + location + " with " + weather_

    except:
        temp = "Could not get results"

    return temp
