# review_IMDB

import requests
import re

url = 'https://www.imdb.com/chart/top/'

received = requests.get(url).text

title = re.findall(r'alt="(.+)"/>', received)
# print(title)

rating = re.findall(r'<strong title=".+">(.+)</strong>', received)
# print(rating)

rank = re.findall(r'<td class="titleColumn">\n\s\s\s\s\s\s(.+)', received)
# print(rank)

year = re.findall(r'<span class="secondaryInfo">(.+)</span>', received)
# print(year)

for rank, title, year, rating in zip(rank, title, year, rating):
    print(rank, title, year, rating)
