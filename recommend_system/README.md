# Recommendation system

> In this folder you will find two python scripts that containes logic and code of working recommendation system and also explonation to them. Also you will find explonation of each step of installation dependencies for running app.
## Setup enviroment:

> Python installation:
- Here is link of explanation how you can install it on mac ios: https://www.mathison.ch/en-ch/blog/so-installieren-sie-python-311-ganz-einfach-auf-ih/
- Here is link of explanation how you can install it on ubuntu/linux system: https://www.thisisckm.com/post/python-3-11-how-to-install-on-ubuntu
- Here is link of explanation how you can install it on Windows system: https://www.datacamp.com/blog/how-to-install-python#
- After installation, you can check your Python version with the following command: python3.11 --version
- Before starting work, you must create a virtual environment where you will install all the necessary packages and libraries. Use the following command to create the environment:*python -m venv venv
- This will create a new folder called venv that contains the Python virtual environment.
> Install dependancies:
- The list of all necessary packages and libraries I provided in **requirements.txt** file.
- So first you need activate your previous created python virtual enviroment by the next coomand: **source venv/bin/activate**
- After you had activated python enviroment you need install all packages and libraries that contain requirements.txt file with the following command: **pip install -r requirements.txt**
- Also in your working folder you must create a file with name: **.env** and inside this file you must create a new variable for openai api key: ***OPENAI_API_KEY = "your key from openai account"***
## Running the Recommendation Apps:

### Overvie: We have two recommendation apps, one without a database and the other with a database. The primary difference is how they handle and retrieve data for generating recommendations.
> ***recommend_nodb.py** This is a simple Streamlit app that works without a database. It's designed to demonstrate how the recommendation system works using pre-trained data without needing any external data input. Use Case: Ideal for showcasing to clients how the system generates recommendations based on user preferences without requiring additional data input.
- So for the starting app you must: Open your terminal. Run the following command: **streamlit run recommend_nodb.app** This will open a new webpage in your browser. In the app window, you can enter your preferences for items like movies, cars, games, products, books. The system will generate and display three similar recommendations based on the trained data.

>**recommend_app_db.py** This app works similarly to the first but with added functionality to upload PDF files. It allows you to search for similar products based on the data contained within the uploaded documents.
- Suitable for scenarios where you have specific data (e.g., product catalogs, book lists) in PDF format, and you want to generate recommendations based on this data.
- So for the starting app you need to write the command: **streamlit run recommend_app_db.py**. Upload PDF files that contain data about products, items, books, etc.. Enter your preferences in the provided input field. The app will process the uploaded documents and generate three similar items based on your dataset.

## Key Points
> recommend_nodb.py:

- No need for external data.
- Demonstrates the recommendation system using pre-trained data.
- Ideal for quick demonstrations.

> recommend_app_db.py:

- Requires PDF file uploads containing relevant data.
- Generates recommendations based on user-provided documents.
- Suitable for more tailored and data-specific recommendations.