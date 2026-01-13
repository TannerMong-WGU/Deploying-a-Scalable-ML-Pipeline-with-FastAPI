import json

import requests

get_url = "http://127.0.0.1:8000/"
r = requests.get(get_url)

# Print status code and welcome message
print("GET request status code:", r.status_code)
print("GET request response:", r.json())



data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

post_url = "http://127.0.0.1:8000/data/"
r = requests.post(post_url, json=data)

# Print status code and prediction result
print("POST request status code:", r.status_code)
print("POST request response:", r.json())
