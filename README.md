# 🚗 **Project: Car-Post Recommender**

## 📝 **Summary**
This project recommends used vehicles listed for sale on Craigslist that are considered **undervalued**. Its purpose is to help users find good deals more easily by avoiding overpriced vehicles or potential scams.

## 🔧 **Process**
Using machine learning, the model predicts a vehicle’s fair market value. The difference between the predicted price and the seller’s asking price determines whether a vehicle is undervalued. An API allows users to view a sorted list of undervalued car posts, with the option to filter by make. Potential scams or illegitimate posts are filtered out by identifying outliers.

## 📋 **Tasks**
- Web scraped Craigslist using **BeautifulSoup** and **Selenium**
- Saved and split datasets into **SQLite**
- Performed **Exploratory Data Analysis** and **Feature Engineering**
- Cleaned and preprocessed data using pipelines
- Built and evaluated a machine learning model
- Created an API using **FastAPI**
- Implemented a **CI/CD pipeline** with **Docker** and **GitHub Actions**
- Deployed the project to **AWS ECS**

## 🐞 **Known Issues**
- The current model unfortunately does **not predict well**, and I haven't yet identified why. I'm aware of this and plan to improve it.
- The Craigslist web scraper occasionally breaks due to **changes in their site structure**. The scraper will need to be updated periodically to stay functional.

## 🔧 **Planned Improvements**
- Collect a larger, more diverse dataset
- Improve frontend design to be more user-friendly and appealing
- Continuously update with the latest car listings
- Analyze and utilize the full text from scraped descriptions using NLP

## ✅ **Conclusion**
This has been one of the most challenging and rewarding projects I've worked on. From learning how to build and clean datasets from scratch, to deploying a working API on AWS using Docker and GitHub Actions, I’ve picked up a wide range of real-world skills. There were plenty of struggles — especially around model performance and deployment — but they pushed me to learn more than I expected. I now feel much more confident in my ability to build end-to-end machine learning systems and deploy them in a professional, scalable way. This project also showed me how messy real-world data can be and gave me a real appreciation for thoughtful data engineering and model evaluation.

## ✅ Proof of Deployment

Below is a screenshot showing the API live on AWS using its public IP:

![Deployed API Screenshot](screenshots/Screenshot(23).png)

![Deployed API Screenshot](screenshots/Screenshot(24).png)

![Deployed API Screenshot](screenshots/Screenshot(25).png)

![Deployed API Screenshot](screenshots/Screenshot(26).png)

