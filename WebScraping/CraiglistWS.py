import requests
from bs4 import BeautifulSoup as bs

HEADERS = {
    "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0"
}


def get_car_url(page_number):
    search_url = f"https://dallas.craigslist.org/search/cta?purveyor=owner#search=1~gallery~{page_number}~0"
    r = requests.get(search_url, headers=HEADERS)
    soup = bs(r.content, 'html.parser')

    car_listing = soup.find('ol', class_='cl-static-search-results')
    car_posts = car_listing.find_all('li', class_='cl-static-search-result')
    car_urls = []
    for car in car_posts[:3]:
        url = car.find('a', href=True)
        if url:
            car_urls.append(url.get('href'))

    return car_urls

def get_car_details(url):
    r = requests.get(url, headers=HEADERS)
    soup = bs(r.content, 'html.parser')
    attr_groups = soup.find_all('div', class_='attrgroup')
    year = soup.find('span', class_='valu year')
    model = soup.find('span', class_='valu makemodel')
    price = soup.find('span', class_="price")
    attributes = {
        "Year":year, 
        "Model":model, 
        "Price":price
        }
    spans = attr_groups[1].find_all('span')
    for i in range(0, len(spans), 2):
        key = spans[i].text.strip()
        if i + 1 < len(spans):
            value = spans[i + 1].text.strip()
            attributes[key] = value
    return attributes

def main():
    # base_url = "https://dallas.craigslist.org/search/cta?purveyor=owner#search=1~gallery~0~0"
    # all_car_urls = []
    # for page_number in range(0,18):
    #     car_urls = get_car_url(page_number)
    #     all_car_urls.extend(car_urls)

    # for url in car_urls:
    #     details = get_car_details(url)
    #     print(details)
    car_urls = get_car_url(0)  # 0 corresponds to the first page
    
    # Check if URLs were retrieved
    if car_urls:
        for url in car_urls:  # Loop through the car URLs
            print(f"Scraping details for: {url}")
            details = get_car_details(url)
            print("Car Details:")
            for key, value in details.items():
                print(f"{key}: {value}")
            print("-" * 40)  # Optional separator between car details
    else:
        print("No car URLs found. Check your scraping logic or the structure of the website.")
if __name__ == "__main__":
    main()
