# Project 1: CRISP-DM Process
This project employs the CRISP-DM method for the exploration, analysis, modeling and interpretation of the results obtained by studying a dataset, specifically [AirBnB](https://www.kaggle.com/airbnb/seattle).

The code was used to answer three important questions (Understanding Data):

1. What are the most common amenities offered at the host sites? how does it affect the price of host?
2. What is the month with the most booked reservations in Seattle, and by what percentage do prices increase?
3. What relates to price?

# Structure of the project
Within the download you'll find the following directories and files

```text
project_crispdm/
├── code.ipynb
├── utils.py
├── requirements.txt
├── dataset/
│   ├── calendar.csv
│   ├── listings.csv
│   ├── reviews.csv
```

`code.ipynb`
Main code; contains all the code that reflects the exploration, analysis and visualization of the data

`dataset`:
- Listings, including full descriptions and average review score
- Reviews, including unique id for each reviewer and detailed comments
- Calendar, including listing id and the price and availability for that day

`utils.py`
Contains utility functions to process part of the information handled in the main code.

`requirements.txt`
All main dependencies

# Prerequisites
Python 3.7

# Getting Started

## Pre-installation:
This process assumes by default that you have python 3 as the default language on your system.

### Install **virtualenv** using pip3
```
pip3 install virtualenv 
```
### Create a virtual environment using Python 3
```
virtualenv -p python3 venv 
```
>Any name other than **venv** can be used.

### With other Python interpreter
```
virtualenv -p <PATH_OTHER_PYTHON_3> venv
```
### Active your virtual environment:    
```  
source venv/bin/activate
```
### To deactivate:
```
deactivate
```

## Download & Install

### Clones the project locally
```
git clone https://github.com/leonellima/project_crispdm.git
```

### Install dependencies
With the virtual environment active, execute the following command
```
pip install -r requirements.txt
```

### Run the notebook with code
```
jupyter notebook code.ipynb
```

# Author
Danny Lima