import requests

def getjoke():
    url = f"https://icanhazdadjoke.com/"

    headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.99 Safari/537.36",
                "Accept": "text/plain"
                }

    req = requests.get(url, headers=headers)

    temp = req.text
    return temp
