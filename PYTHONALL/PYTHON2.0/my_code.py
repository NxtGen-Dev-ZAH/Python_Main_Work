def calculate_area_of_rectangle(length, width) :
    return length*width;

def calculate_area_of_triangle(base,height) :
    return 1/2*(base*height);

def calculate_average_of_three_numbers(numbers):
    return sum(numbers)/len(numbers);


# from bs4 import BeautifulSoup

# html="<!DOCTYPE html><html><head><title>STATUS OF PLAYERS </title></head><body><h3><b id='boldest'>LebronJames</b></h3><p> Salary: $ 92,000,000 </p><h3> Stephen Curry<h3><p› Salary: $85,000, 000 </p›<h3> Kevin Durant </h3><p> Salary: $73,200, 000</p></body></html>"
# soup = BeautifulSoup(html, 'html5lib')

# tag_object=soup.title
# print(tag_object)
# tag_object2=soup.h3
# tagchild=tag_object2.b
# tagparent=tagchild.parent
# print(tagchild)
# print(tagparent)
# print(tagchild.attrs)
# print(tagchild.string)

# from bs4 import BeautifulSoup
# import requests
# url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DA0321EN-SkillsNetwork/labs/datasets/HTMLColorCodes.html"

# data=requests.get(url).text

# belement=BeautifulSoup(data,'html.parser')

# table=belement.find('table')
# #Get all rows from the table
# for row in table.find_all('tr'): # in html table row is represented by the tag <tr>
#     # Get all columns in each row.
#     cols = row.find_all('td') # in html a column is represented by the tag <td>
#     color_name = cols[2].string # store the value in column 3 as color_name
#     color_code = cols[3].string # store the value in column 4 as color_code
#     print("{}--->{}".format(color_name,color_code))



# import pandas as pd
# url = "https://en.wikipedia.org/wiki/World_population"
# data  = requests.get(url).text
# soup = BeautifulSoup(data,"html.parser")
# tables = soup.find_all('table')
# len(tables)
# for index,table in enumerate(tables):
#     if ("10 most densely populated countries" in str(table)):
#         table_index = index
# print(table_index)
# print(tables[table_index].prettify())
# population_data = pd.DataFrame(columns=["Rank", "Country", "Population", "Area", "Density"])

# for row in tables[table_index].tbody.find_all("tr"):
#     col = row.find_all("td")
#     if (col != []):
#         rank = col[0].text
#         country = col[1].text
#         population = col[2].text.strip()
#         area = col[3].text.strip()
#         density = col[4].text.strip()
#         population_data = population_data.append({"Rank":rank, "Country":country, "Population":population, "Area":area,     "Density":density}, ignore_index=True)

# population_data