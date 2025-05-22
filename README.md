# **Project: Car-Post Recommender**

## **Summary:**
This project recommends used vehicles being posted for sale on Craigslist that are considered undervalued. It's purpose is to help people find a undervalued vehicle to buy, making the search process easier by avoiding overpriced vehicles or potential scams.

## **Process:**
With the use of machine-learning, the built model predicts the price of each vehicle. The difference between the predicted price and the asking price of the vehicle being sold is used to determine if the car is undervalued. An API is provided for the user to be able to view a listing of car posts sorted by the most undervalued vehicles. The user is also allowed to filter by the make of the vehicle they prefer to view. The project avoids potential scams or car posts that are non-legitimate by identifing them as outliers and removing them.





## What was done
* Webscaped Craigslist with the use of Beatifulsoup and Selinium
* Saved and split datasets into SQLite
* Performed EDA and Feature Engineering
* Cleaned and Preprocessed Data with Pipeline
* Model building
* Created API using FastAPI
* CI/CD Pipeline with Docker and GitHub Actions (Deployed to AWS ECS)


## important notes

## **Known issues:**
The biggest issue that i regret to mention is that the current model built unfortionately doesn't predict well. I'm not sure what the issue is, but I'm aware of it and plan to figure it out as soon as I can.

Another issue I'm aware of is that the Webscraper program isn't up to date with Craigslist website, as their website changes every now and then the webscraper will need to be updated accordingly.


## **Improvements to be made:**
* Collect more data
* Frontend design to be more appealing
* Continously update the lastest car-posts
* Make use of the scraped description


## **Conclustion:**