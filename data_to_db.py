import csv
from pymongo import MongoClient

HOST = 'cluster0.ewxae.mongodb.net'
USER = 'nils'
PASSWORD = 'nils0901'
DATABASE_NAME = 'myFirstDatabase'
COLLECTION_NAME = 'london_bike'
MONGO_URI = f"mongodb+srv://{USER}:{PASSWORD}@{HOST}/{DATABASE_NAME}?retryWrites=true&w=majority"

client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

with open('london_bike.csv', 'r') as f:
        csv_reader = csv.DictReader(f)
        for i in csv_reader:
            i["cnt"] = int(i["cnt"])
            i["t1"] = float(i["t1"])
            i["t2"] = float(i["t2"])
            i["hum"] = float(i["hum"])
            i["wind_speed"] = float(i["wind_speed"])
            i["weather_code"] = float(i["weather_code"])
            i["is_holiday"] = float(i["is_holiday"])
            i["is_weekend"] = float(i["is_weekend"])
            i["season"] = float(i["season"])

            collection.insert_one(document=i)
