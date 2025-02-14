import requests
from bs4 import BeautifulSoup as bs
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from requests.exceptions import HTTPError
import csv
import random
import re
import logging
import os

user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0",
    "Mozilla/5.0 (Linux; Android 14; SM-A037U1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 [ip:181.230.212.104]",
    "Mozilla/5.0 (Linux; diordnA 14; SM-X820 Build/UP1A.231005.007; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/132.0.6834.56 Safari/537.36",           
    "Mozilla/5.0 (Linux; Android 13; SM-T505) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.6635.63 Mobile Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_3_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) FxiOS/131.2 Mobile/15E148 Safari/605.1.15 Chrome/128.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0.9082) Gecko/20100101 Firefox/133.0.9082",
    "Mozilla/5.0 (X11; U; Linux i686; en-US; rv:101.0) Gecko/20172119 Firefox/101.0"
]



seen_urls = set()
unique_field_names = set()
search_querry = ["dallas","austin","houston"]
logging.basicConfig(filename='scraping_log.txt', 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def get_car_url(location):
    search_url = f"https://{location}.craigslist.org/search/cta?purveyor=owner#search=1~gallery~0~0"
    logging.info(f"search url:{search_url}")
    PATH = r"C:\Program Files (x86)\chromedriver.exe"
    service = Service(executable_path=PATH)
    options = webdriver.ChromeOptions()
    options.add_argument(f"user-agent={random.choice(user_agents)}")
    driver = webdriver.Chrome(service=service, options=options)
    car_urls = []
    backoff_factor = 3
    driver.get(search_url)
    time.sleep(5)
    page_num = 0
    
    while True:
        try:
            result = driver.page_source
            soup = bs(result,'html.parser')
            WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CLASS_NAME, "cl-search-result")))
            car_listing = soup.find('div', class_="results cl-results-page")
            car_posts = car_listing.find_all('div', class_='cl-search-result cl-search-view-mode-gallery')

            logging.info(f"Scraping [page {page_num}]: Found {len(car_posts)} posts")
            for car in car_posts: # test: obtain only 3 car postings using "for car in car_posts[:3]"
                url = car.find('a', href=True)
                if url and url.get('href'):
                    car_urls.append(url.get('href'))  

            try:
                next_page_btn = WebDriverWait(driver, 30).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, ".bd-button.cl-next-page.icon-only"))
                )
                
                time.sleep(random.uniform(2,4))

                next_page_btn = driver.find_element(By.CSS_SELECTOR, ".bd-button.cl-next-page.icon-only")
                if "bd-disabled" in next_page_btn.get_attribute("class"):
                    print("Reached last page. Stopping.")
                    break
                next_page_btn.click()
                time.sleep(random.uniform(45, 90))
                WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.CLASS_NAME, "cl-search-result")))

                result = driver.page_source
                soup = bs(result,'html.parser')

            except NoSuchElementException as e:
                logging.error(f"Page {page_num}: No such element found. Error: {e}. Stopping page scraping.")
                continue
            page_num+=1
            
        except HTTPError as e:
            if e.response.status_code == 410:
                logging.warning(f"410 Error: {e}. URL is gone. Skipping...")
                break
            elif e.response.status_code == 412:
                logging.error(f"Precondition Failed (412): {e}. Skipping URL.")
                break
            elif e.response.status_code == 429:
                logging.warning(f"Rate limiting (429): Too Many Requests. Retrying later...")
                wait_time = random.uniform(600, 1200)
                logging.info(f"Waiting for {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            else:
                wait_time = backoff_factor ** 100
                logging.warning(f"HTTP error: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
        except Exception as e:
            logging.error(f"Failed to get product links on page: {page_num}. Error: {e}")
            continue

    return car_urls

def get_car_details(url):
    max_retries = 5
    backoff_factor = 3
    attributes = {}
    for attempt in range(max_retries):
        HEADERS = {"User-Agent":random.choice(user_agents)}
        try:
            r = requests.get(url, headers=HEADERS)
            r.raise_for_status()
            soup = bs(r.content, 'html.parser')
            attr_groups = soup.find_all('div', class_='attrgroup')
            year = soup.find('span', class_='valu year')
            model = soup.find('span', class_='valu makemodel')
            price = soup.find('span', class_="price")
            post_id = url.split("/")[-1].replace(".html", "")
            description_section = soup.find('section', id= 'postingbody')
            if description_section:
                unwanted_p = description_section.find('p')
                unwanted_a = description_section.find('a')
                if unwanted_p:
                    unwanted_p.decompose()
                if unwanted_a:
                    unwanted_a.decompose()
                description = description_section.text.strip()
                description = re.sub(r'\s+', ' ', description)
            else:
                description = "[No Description Provided]"
            
            attributes = {
                "Year":year.text if year else None, 
                "Model":model.text if model else None, 
                "Price":price.text if price else None,
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
                    value = attributes.get(key)
                    attributes[key] = clean_data(value) if value else None
        except HTTPError as e:
            if e.response.status_code == 410:
                logging.warning(f"410 Error: {e}. URL is gone. Skipping...")
                break
            elif e.response.status_code == 412:
                logging.error(f"Precondition Failed (412): {e}. Skipping URL.")
                break
            elif e.response.status_code == 429:
                logging.warning(f"Rate limiting (429): Too Many Requests. Retrying later...")
                wait_time = random.uniform(600, 1200)
                logging.info(f"Waiting for {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            else:
                wait_time = backoff_factor ** attempt
                logging.warning(f"HTTP error: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
        except Exception as e:
            logging.error(f"Failed to get details for {url}. Error: {e}")
            continue

    return attributes
        
def initialize_csv():
    with open("Houston&AustinDataset.csv", 'w', newline='', encoding='utf-8') as f:
       pass
    print('CSV created')

def append_to_csv(page_details):
    field_names = list(unique_field_names)
    file_path = "Houston&AustinDataset.csv"

    # Check the file size before reading
    if os.path.getsize(file_path) == 0:  # File is empty
        with open(file_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=field_names)
            writer.writeheader()
    with open("Houston&AustinDataset.csv", 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writerows(page_details)

def clean_data(value):
    rm_chars = ["$",',']
    for char in rm_chars:
        if char in value:
            value = value.replace(char, "")
    return value

def main():
    # base_url = "https://dallas.craigslist.org/search/cta?purveyor=owner#search=1~gallery~0~0"
    batch_size = 600
    
    logging.info("Started scraping...")
    initialize_csv()
    for location in search_querry:
        logging.info(f"Scraping pages for {location}...")
        car_urls = get_car_url(location)
        if not car_urls:
            logging.info(f"No car URLs found for {location}. Stopping.")
            break
        else:
            logging.info(f"Found {len(car_urls)} car URLs for {location}.")
        counter = 1
        page_details = []
        for url in car_urls:
            if len(page_details) >= batch_size:
                append_to_csv(page_details)  
                logging.info(f"Saved {len(page_details)} listings from {location}")
                page_details.clear()
            if url not in seen_urls:
                details = get_car_details(url)
                if details:
                    print(f"Instance: {counter}")
                    print(details)
                    print("-" * 40)
                    page_details.append(details)
                else:
                    logging.warning(f"Skipping empty details for {url}")
                    continue
                seen_urls.add(url)  
                counter+=1  
                time.sleep(random.uniform(5, 10))
            
        if page_details:
            append_to_csv(page_details)  
            logging.info(f"Saved {len(page_details)} listings from {location}")
            
           
    logging.info("Finished scraping.")
    
if __name__ == "__main__":
    main()
