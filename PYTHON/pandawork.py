import pandas as pd
# print("panda workings here")
# df=pd.read_csv('datacs.csv',skiprows=[16])
# # print(df.to_string())
# mydataset={
#     "cars"    : ["BMW" ,"VOLVO","FORD"],
#     "PASSINGS": ["best","worst","good"],
#     "POWER"   : ["1200HP","600HP","800HP"]
# }
# MYVAR=pd.DataFrame(mydataset)
# print(MYVAR," \n ")
# print(MYVAR.loc[0],"\n")
# # print(MYVAR.loc[[1,2]] ," \n ")
# df = pd.DataFrame(mydataset, index = ["PETROL 1", "PETROL 2", "PETROL 3"])
# print(df," \n ")
# print(df.loc["PETROL 2"])


# a=[1,7,34,56,23,67]#THIS IS THE SERIES WITHOUT LABELS
# myvar=pd.Series(a)
# print(myvar[0])
# print(myvar)
# MYvar=pd.Series(a,index=["x","y","z","b","a","c"])
# print(MYvar["b"]) #IT WILL RETURN ONLY NUMBER

# calories = {"day1": 420, "day2": 380, "day3": 390}
# pandas45 = pd.Series(calories,index=["day1","day2"])
# #IT WILL RETURN LABELS AND KEYVALUE AS WELL

# print(pandas45)


data = {
  "Pulse":
  {
    "0":110,
    "1":117,
    "2":103,
    "3":109,
    "4":117,
    "5":102
  },
  "Maxpulse":
  {
    "0":130,
    "1":145,
    "2":135,
    "3":175,
    "4":148,
    "5":127
  },
  "Calories":
  {
    "0":409,
    "1":479,
    "2":340,
    "3":282,
    "4":406,
    "5":300
  }
}

df = pd.DataFrame(data)
print(df["Calories"])

# df=pd.read_csv('datacs.csv')
# print(df[df['strike rate']>=400])
# newdf=df.dropna()
# print(newdf.to_string())
# df.dropna(inplace = True)
# print("this is the oringinal dataframe")
# print(df.to_string())
# a=df["strike rate"].mean()
# df["strike rate"].fillna(a,inplace= True)# it will work on one column empty values
# data = [10, 15, 20, 25, 30]
# series = pd.Series(data)
# mean_value = series.mean()
# print("Mean:", mean_value)

# import pandas as pd
# dataset={'A':[120,34,32,120,45,45],
#          'B':[25,6,7,89,65,6],
#          'C':(26,47,89,76,pd.NaT,47)}
# dfa=pd.DataFrame(dataset)
# # meanvalues=dfa.mean()
# # dfa.loc[5,'B']=45
# # print(dfa)
# # print("this is the mean value of the whole table",meanvalues)
# # mn=dfa['A'].mode()
# # NM=dfa.mode()
# # print("this is the mode of a column",mn)
# # print("this is the mode of whole table ",NM)
# # dfa.dropna(subset=['C'],inplace=True) # drop null values in column C
# # print(dfa)
# count=0
# for x in dfa.index:
#     if dfa.loc[x, "A"] == 120:
#        count=count+1
      
# print(count)

#mean is the average of values a.mean() b=[23,4,45,56] a=pd.series(b)
#.DataFrame(),.Series(),.read_csv( ),.read_json(),.tail(),.head(),.info()
#.dropna(),.fillna(value,inplace=TRUE),.loc[index],df[['NAME','AGE']]
#.median(),
#The head() method returns the headers and a specified number of rows,
#tail() method for viewing last row of dataframe



import pandas as pd

dataset = {'A': [120, 34, 32, 120, 45, 45,32],
           'B': [25, 6, 7, 89, 65, 6,7],
           'C': [26, 47, 89, 76, 0, 47,89]}

dfa = pd.DataFrame(dataset,dtype=int)
print(dfa)
print(dfa.corr())
#this .corr() prints corelation between the tables 
# Convert column "A" to numeric (if it's not already)
#dfa['A'] = pd.to_numeric(dfa['A'], errors='coerce')

count = 0
# for x in dfa.index:
#     if dfa.loc[x, "A"] > 120.00:
#         count +=1
count1=0
for index, row in dfa.iterrows():
    if row['C'] > 40:
        count1 += 1

print(dfa.duplicated())
#dfa.drop_duplicates(inplace=True)

for index,row in  dfa.iterrows():
    if row["A"] > 110:
        dfa.drop(index , inplace= True)
        count+=1

print("this is the amount of items greater than 110 in A",count1)
print(dfa['A'])
print("this is the no of the items deleted",count)

import pandas as pd
DFAN=pd.read_csv('datacsv2.csv')
#print(DFAN.head())
# print(DFAN.iloc[3])
# print(DFAN.loc[2])
# print(DFAN.iloc[0,0])
print(DFAN.loc[4,'Maxpulse'])
print(DFAN.iloc[4,3])
print(DFAN.loc['2020/12/03',2])

n=pd.DataFrame(DFAN)
print(n.iloc[0,0])
m=n[['Duration','Maxpulse']]
m.to_csv('demofile.csv',index=False)
p=m["Duration"]>45#this will only return boolean values either true or false
#the column name such as maxpulse(million) will be maxpulse(million) not maxpulse
print(p)
p=m[m["Duration"]>= 45]#this will return values 
print(p)
p.to_csv('demofile.csv')
print(m['Duration'].unique())
print(m)
#print(n)


# from pycoingecko import CoinGeckoAPI
import pandas as pd
import matplotlib.pyplot as plt
# cg=CoinGeckoAPI()
# bitcoin=cg.get_coin_market_chart_by_id(id='bitcoin',vs_currency='usd',days=30)
# print(bitcoin)
# data=pd.DataFrame(bitcoin,columns=['prices'])
# print(data)
# # Create a DataFrame with the prices

# # Split the price column into two columns
# price_column = data['prices']
# timestamps=[]
# for price in price_column:
#     timestamps.append(pd.to_datetime(price[0], unit='ms'))
# pricess=[]
# for price2 in price_column:
#     pricess.append(pd.to_numeric(price2[1]))

# # Create a new DataFrame with the split columns
# datass = pd.DataFrame({'timestamp': timestamps,'PRICE':pricess})
#datass.to_csv('demofile3.csv',index=False)

datas=pd.read_csv('demofile3.csv')
datas.plot()
plt.show()
#plt.plot where there a two points in program ,pd.read_csv=m so m.plot()
# Print the new DataFrame
#print(data)
print("this is the endline")


# Import the necessary libraries

import pandas as pd
import numpy as np
df=pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['a', 'b', 'c'])
print(df)
dfn = df.transform(func = lambda x : x + 10)
print(dfn)
result = dfn.transform(func = ['sqrt'])
print(result.astype(int))

# import pandas as pd
# import pyfetch 
# filename = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0101EN-SkillsNetwork/labs/Module%205/data/diabetes.csv"

# async def download(url, filename):
#     response = await pyfetch(url)
#     if response.status == 200:
#         with open(filename, "wb") as f:
#             f.write(await response.bytes())

# await download(filename, "diabetes.csv")
# df = pd.read_csv("diabetes.csv")