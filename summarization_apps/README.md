# Summarization system

> In this folder you will find two python scripts that containes logic and code of working summarization system and also explonation to them. Also you will find explonation of each step of installation dependencies for running app.
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
## Running the Summarization Apps:

### Overvie: We have two summarization apps. One provides summaries based solely on your uploaded document, while the other generates summaries based on a client's prompt.
> ***summarization_app.py** This app provides summaries of uploaded files by extracting the main points. It demonstrates how the summarization system works using the uploaded data. The process is divided into two parts: first, it splits your document into chunks, then it generates brief summaries for each chunk, combines all the chunk summaries into one document, and finally provides the final summarization.

Running the App:
- Open your terminal.
- Run the following command: **streamlit run summarization_app.py**
- This will open a new webpage in your browser.
- In the app window, you can select the chunk size and overlap parameters to control how the document is split into chunks.
- Upload your PDF file.
- Press Enter.
- After processing, you will see the final summarization of your document.

>**summarization_app_2.py** This summarization app works slightly differently by providing summaries based on client prompts. You can define specific prompts to tailor the summarization process.

Running the App:
- Open your terminal.
- Run the following command: **streamlit run summarization_app_2.py**
- This will open a new webpage in your browser.
- Define the chunk size and overlap strategies.
- Upload your PDF file.
- Define your prompt (e.g., "Provide a brief summarization of the following document and list the main points").
- Press Enter.
- The app will generate a summarization based on your prompt and the uploaded document.

## Key Points
> summarization_app.py:
- Summarizes uploaded documents by splitting them into chunks and combining summaries of each chunk.
- Suitable for general document summarization without additional context.

> summarization_app_2.py:
- Allows for custom prompts to tailor the summarization process.
- Ideal for specific summarization needs based on user-defined prompts.