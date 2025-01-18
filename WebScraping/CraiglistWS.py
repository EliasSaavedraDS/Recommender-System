import requests
from bs4 import BeautifulSoup as bs

HEADERS = {
    "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0"
}
URL = "https://books.toscrape.com/"


def get_book_url(url):
    r = requests.get(url, headers=HEADERS)
    soup = bs(r.content, 'html.parser')
    #listing = soup.find_all('article', class_='product_pod')
    links = soup.find_all('a', href=True)
    book_urls = []
    for book in links:
        book_url = book['href']
        if book_url.startswith('https'):
            book_url = URL + book_url
        else:
            book_urls.append(book_url)
    return book_urls

def get_book_details(book_url):
    r = requests.get(book_url, headers=HEADERS)
    soup = bs(r.content, 'html.parser')
    title = soup.find('h1').text
    price = soup.find('p', class_='price_color').text
    stock = soup.find('p',class_='instock availability').text.strip()
    return {
        "Title":title,
        "Price":price,
        "Stock":stock
    }
    
        


    


book_urls = get_book_url(URL)

for url in book_urls:
    details = get_book_details(url)
    print(details)