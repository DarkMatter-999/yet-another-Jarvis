import requests, json
from bs4 import BeautifulSoup

def google_search(q):
    search = q.replace(" ", "+")

    url = f"https://www.google.com/search?&q={search}"

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.99 Safari/537.36"}
    req = requests.get(url, headers=headers)
    sor = BeautifulSoup(req.text, "html.parser")

    try:
        temp = sor.find("div", class_='wDYxhc NFQFxe viOShc LKPcQc').text
    except:
        try:
            temp = sor.find("span", class_='ILfuVd').text
        except:
            try:
                temp = sor.find("div", class_='Z0LcW').text
            except:
                try:
                    temp = sor.find("div", class_='ayRjaf').text
                except:
                    temp = "Could not get results"

    return temp
