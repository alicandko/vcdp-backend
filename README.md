# VCDP-backend
This is an API for dataset preparation and video classification.  
## Prerequisites
Install virtualenv  
```
pip install virtualenv
```
## Installing
Go to directory  
```
cd vcdp-backend
```
Create virtual environment  
```
virtualenv env
```
Activate virtual environment  
```
source env/bin/activate
```
Install dependencies  
```
pip install -r requirements.txt
```
## Setup
Go to https://console.developers.google.com and create a project.  
Get an API key and enter it to the respective field at vcdp/videos/settings.py.  
```
DEVELOPER_KEY = "your key goes here"
```
Get an OAuth client ID, copy and paste the content of the json file to vcdp/google_application_credentials.json  
## Running the tests
Go to directory  
```
cd vcdp
```
To run the tests  
```
python manage.py test
```
## Running the app
Go to directory  
```
cd vcdp
```
To run the app  
```
python manage.py runserver
```
The Browsable API will be available at http://localhost:8000/