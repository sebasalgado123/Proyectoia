import requests

url = "http://127.0.0.1:5000/clasificar"
reseña = {"texto": "The movie was absolutely fantastic and thrilling!"}

response = requests.post(url, json=reseña)
print(response.json())
