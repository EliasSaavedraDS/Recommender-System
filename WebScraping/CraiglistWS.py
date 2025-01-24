import requests
from bs4 import BeautifulSoup as bs
import time
from requests.exceptions import HTTPError
import csv
import random
import re

HEADERS = {
    "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0"
}

seen_urls = set()
unique_field_names = set()

def get_car_url(page_number):
    search_url = f"https://dallas.craigslist.org/search/cta?purveyor=owner#search=1~gallery~{page_number}~0"
    car_urls = []
    max_retries = 5
    backoff_factor = 3
    for attempt in range(max_retries):
        try:
            r = requests.get(search_url, headers=HEADERS)
            r.raise_for_status()
            soup = bs(r.content, 'html.parser')
            car_listing = soup.find('ol', class_='cl-static-search-results')
            car_posts = car_listing.find_all('li', class_='cl-static-search-result')
            for car in car_posts: # test: obtain only 3 car postings with-> for car in car_posts[:3]
                url = car.find('a', href=True)
                if url and url.get('href'):
                    car_urls.append(url.get('href'))    
            break
        except HTTPError as e:
            if e.response.status_code == 412:
                print(f"Precondition Failed (412): {e}. Skipping URL.")
                break
            wait_time = backoff_factor ** attempt
            print(f"HTTP error: {e}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"Failed to get product links on page: {page_number}. Error: {e}")
            break
    return car_urls

def get_car_details(url):
    max_retries = 5
    backoff_factor = 3
    attributes = {}
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=HEADERS)
            r.raise_for_status()
            soup = bs(r.content, 'html.parser')
            attr_groups = soup.find_all('div', class_='attrgroup')
            year = soup.find('span', class_='valu year').text
            model = soup.find('span', class_='valu makemodel').text
            price = soup.find('span', class_="price").text
            post_id = url.split("/")[-1].replace(".html", "")
            description_section = soup.find('section', id= 'postingbody')
            if description_section:
                unwanted_div = description_section.find('div')
                unwanted_a = description_section.find('a')
                if unwanted_div:
                    unwanted_div.decompose()
                if unwanted_a:
                    unwanted_a.decompose()
                description = description_section.text.strip()
                description = re.sub(r'\s+', ' ', description)
            else:
                description = "[No Description Provided]"
            
            attributes = {
                "Year":year, 
                "Model":model, 
                "Price":price,
                "Description":description,
                "Post ID":post_id
                }
            spans = attr_groups[1].find_all('span')
            for i in range(0, len(spans), 2):
                key = spans[i].text.strip()
                if i + 1 < len(spans):
                    value = spans[i + 1].text.strip()
                    attributes[key] = value

            for key in attributes.keys():
                unique_field_names.add(key)

            for key in unique_field_names:
                attributes[key] = attributes.get(key, None)

            for key in attributes:
                if key != "Description":
                    attributes[key] = clean_data(attributes[key])

        except HTTPError as e:
            if e.response.status_code == 412:
                print(f"Precondition Failed (412): {e}. Skipping URL.")
                break
            wait_time = backoff_factor ** attempt
            print(f"HTTP error: {e}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"Failed to get details for {url}. Error: {e}")
            break
    return attributes
        
def export_to_csv(all_details):
    field_names = list(unique_field_names)
    with open("CrailistDataset.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(all_details)
    print('Saved to csv')

def clean_data(value):
    rm_chars = ["$",',']
    for char in rm_chars:
        if char in value:
            value = value.replace(char, "")
    return value

def main():
    # base_url = "https://dallas.craigslist.org/search/cta?purveyor=owner#search=1~gallery~0~0"
    
    #block used to export all car details of all car posts for each page 0-17
    all_car_urls = []
    all_details = []
    for page_number in range(0,18):
        car_urls = get_car_url(page_number)
        all_car_urls.extend(car_urls)
        if car_urls:
            for url in car_urls:
                if url not in seen_urls:
                    details = get_car_details(url)
                    if details:  # Ensure we don't append empty data
                        all_details.append(details)
                    else:
                        print(f"Skipping empty details for {url}")
                    #all_details.append(details)
                    seen_urls.add(url) 
                    time.sleep(random.uniform(0.5, 2.0)) 
        else:
            print("No car URLs found. Check your scraping logic or the structure of the website.")
    export_to_csv(all_details)
    
    #Block of code to use to test and print the car details for only the first page. used to see if it is scraping properly
    # car_urls = get_car_url(0)
    # all_details = []
    # if car_urls:
    #     for url in car_urls:
    #         if url not in seen_urls:
    #             print(f"Scraping details for: {url}")
    #             details = get_car_details(url)
    #             all_details.append(details)
    #             print("Car Details:")
    #             time.sleep(0.5)
    #             for key, value in details.items():
    #                 print(f"{key} {value}")
    #             print("-" * 40)  # Optional separator between car details
    #             seen_urls.add(url)  
    # else:
    #     print("No car URLs found. Check your scraping logic or the structure of the website.")
if __name__ == "__main__":
    main()
