---
title: "Natural Disaster and Oil Refinenry Location Visualization"
authors: "Xinyu Zhang"
date: 2019-04-02
output:
  html_document:
    keep_md: true
---



## Project description:

Visualize historical natural disasters and oil refineries the United States on map.

## Approaches:

I first downloaded 60 years of natural disaster data from Federal Emergency Management Agency's [website](https://www.google.com/search?q=fema&oq=fema&aqs=chrome..69i57j69i60l4j35i39.1254j0j7&sourceid=chrome&ie=UTF-8). Then I scraped all crude oil refineries information from Global Energy Observatory's [website](http://globalenergyobservatory.org/list.php?db=Resources&type=Crude_Oil_Refineries). I filtered and cleaned up the data before import into Tableau. Finally, I visualized the data in Tableau.

## Results:

Click picture title to open interactive Tableau Public page.

[60 Years of Natural Disasters in the lower 48 States of US](https://public.tableau.com/views/NaturalDisasterandOilRefinerieLocation/NaturalDisasters?:embed=y&:display_count=yes&publish=yes)
{{< figure library="1" src="natural disaster and oil refinery visualization 1.PNG">}}

[Natural Disasters and Oil Refineries](https://public.tableau.com/views/NaturalDisasterandOilRefinerieLocation/OilRefineriesandNaturalDisasters?:embed=y&:display_count=yes&publish=yes)
{{< figure library="1" src="natural disaster and oil refinery visualization 2.PNG">}}

[States with Oil Refineries](https://public.tableau.com/views/NaturalDisasterandOilRefinerieLocation/OilRefinery-States?:embed=y&:display_count=yes&publish=yes)
{{< figure library="1" src="natural disaster and oil refinery visualization 3.PNG">}}

[States with ExxonMobil's Oil Refineries](https://public.tableau.com/views/NaturalDisasterandOilRefinerieLocation/ExxonMobilRefinery-States?:embed=y&:display_count=yes&publish=yes)
{{< figure library="1" src="natural disaster and oil refinery visualization 4.PNG">}}

[The Top 4 most Frequent Disasters in the States with ExxonMobil's Oil Refineries](https://public.tableau.com/views/NaturalDisasterandOilRefinerieLocation/DisastermostMultiMaps?:embed=y&:display_count=yes&publish=yes)
{{< figure library="1" src="natural disaster and oil refinery visualization 5.PNG">}}

[The Bottom 4 most Frequent Disasters in the States with ExxonMobil's Oil Refineries](https://public.tableau.com/views/NaturalDisasterandOilRefinerieLocation/DisasterleastMultiMaps?:embed=y&:display_count=yes&publish=yes)
{{< figure library="1" src="natural disaster and oil refinery visualization 6.PNG">}}

[Story](https://public.tableau.com/views/NaturalDisasterandOilRefinerieLocation/Story?:embed=y&:display_count=yes&publish=yes)
{{< figure library="1" src="natural disaster and oil refinery visualization 7.PNG">}}

## Python web crawler code:


```rcpp
import csv
import requests
from bs4 import BeautifulSoup

url='http://globalenergyobservatory.org/list.php?db=Resources&type=Crude_Oil_Refineries'
response=requests.get(url)
html=response.content
soup=BeautifulSoup(html,'lxml')
refinery_name_bs_list=soup.find_all('a',attrs={'target':'_blank'})
refinery_link_list=[str(x)[9:str(x).find('target="_blank"')-2] for x in refinery_bs_list]
refinery_name_list=[str(x)[str(x).find('target="_blank"')+16:str(x).find('</a>')-1] for x in refinery_bs_list]

location_list=[]
latitude_list=[]
longitude_list=[]
for x in refinery_link_list:
    url='http://globalenergyobservatory.org/'+x
    response=requests.get(url)
    html=response.content
    soup=BeautifulSoup(html,'lxml')
    refinery_loc_str=str(soup.find('div',attrs={'class':'input_block','id':'Abstract_Block'}))
    location_string=refinery_loc_str[refinery_loc_str.find('is located at')+14:refinery_loc_str.find('Location coordinates are')-2]
    latitude_string=refinery_loc_str[refinery_loc_str.find('Latitude=')+10:refinery_loc_str.find('Longitude=')-2]
    longitude_string=refinery_loc_str[refinery_loc_str.find('Longitude=')+11:refinery_loc_str.find('This infrastructure is')-2]
    location_list.append(location_string)
    latitude_list.append(latitude_string)
    longitude_list.append(longitude_string)
    print(location_string)

location_list_clean=[]
for x in location_list:
    if x[0:2]==', ':
        location_list_clean.append(x[2:])
    else:
        location_list_clean.append(x)

with open('oil refineries.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for i in range(len(refinery_name_list)):
        row=[]
        row.append('http://globalenergyobservatory.org/'+refinery_link_list[i])
        row.append(refinery_name_list[i])
        row.append(location_list_clean[i])
        row.append(latitude_list[i])
        row.append(longitude_list[i])
        writer.writerow(row)
```


