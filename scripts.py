import csv
import requests
import json

csv_file='data.csv'
json_file='output.json'
api_key='AIzaSyDGUJpEFQ98LphcrgqYhilNWmaJRFgf6h0'
result=[ ]
with open(csv_file, newline='',encoding='utf-8') as f:
    reader=csv.DictReader(f)
    for row in reader:
        address=row['address']
        url=f'https://maps.googleapis.com/maps/api/geocode/json?address={address}&key=AIzaSyDGUJpEFQ98LphcrgqYhilNWmaJRFgf6h0'
        response=requests.get(url)
        data=response.json()
        result.append({'address':address,'geocode':data})
with open(json_file,'w',encoding='utf-8') as f:      
    json.dump(result,f, ensure_ascii=False, indent=4)
    print("adresler islendi ve output.json a kaydedildi")


        
