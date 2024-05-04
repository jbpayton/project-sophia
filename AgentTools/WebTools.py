import html2text
import requests

def FetchTextFromURL(url):
    """Fetches content of a URL"""
    response = requests.get(url)
    html_content = response.text

    h = html2text.HTML2Text()
    h.ignore_links = False
    text = h.handle(html_content)
    return text
