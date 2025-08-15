import requests

# Ergast API URL for the latest completed race
url = "https://api.jolpi.ca/ergast/f1/current/last/results.json"

# Make a GET request to the API
response = requests.get(url)
data = response.json()

# Navigate through JSON to find the race winner
race = data['MRData']['RaceTable']['Races'][0]
winner_info = race['Results'][0]['Driver']

winner_name = f"{winner_info['givenName']} {winner_info['familyName']}"
race_name = race['raceName']
race_date = race['date']

print(f"🏎️ Latest Race: {race_name} ({race_date})")
print(f"🥇 Winner: {winner_name}")
