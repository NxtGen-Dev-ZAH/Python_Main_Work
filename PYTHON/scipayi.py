# from scipy import constants
# print(dir(constants))
# print(constants.Avogadro)

# from scipy.optimize import root
# from math import cos

# def eqn(x):
#   return x + cos(x)

# myroot = root(eqn, 0)

# print(myroot.x)

# from scipy import stats
# speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]
# x = stats.mode(speed)
# print(x)

# import numpy as np 
# sea=np.array([32,111,138,28,59,77,97])
# print(np.mean(sea))
# print(np.std(sea))

# # normal is used for gausian distribution mean,standard deviation , size .

# ages = [5,31,43,48,50,41,7,11,15,39,80,82,32,2,8,6,25,36,27,61,31]
# print(np.percentile(ages,75))

'''
import numpy
import matplotlib.pyplot as plt
x = numpy.random.uniform(0.0, 5.0, 100)
y= numpy.random.normal(140,10,100)
plt.hist(x,10)
plt.show()
plt.hist(y,5)
plt.show()
#here in hist(y,5) 5 means no of bars in hist figure
'''

#==========================================================================

# # predict x is year of cars and y is speed 
# import matplotlib.pyplot as plt
# from scipy import stats
# x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
# y = [99,86,87,88,111,86,103,87,94,78,77,85,86]
# slope, intercept, r, p, std_err = stats.linregress(x, y)
# print(r)# here r is corelation coefficient 
# print(p)#p is pobability measure 
# print(std_err)
# def myfunc(x):
#   return slope * x + intercept

# speed = myfunc(10)
# print( "THIS IS THE SPEED OF THE CAR ",speed)
# mymodel = list(map(myfunc, x))
# print(mymodel)
# plt.scatter(x, y)
# plt.plot(x, mymodel)
# plt.show()



#====================================================================

# # Python code explaining
# # numpy.poly1d()
# # importing libraries

# import numpy as np
# # Constructing polynomial
# p1 = np.poly1d([1, 2])
# p2 = np.poly1d([4, 9, 5, 4])

# print ("P1 : ", p1)
# print ("\n p2 : \n", p2)

# # Solve for x = 2
# print ("\n\np1 at x = 4 : ", p1(4))
# print ("p2 at x = 5 : ", p2(5))

# # Finding Roots
# print ("\n\nRoots of P1 : ", p1.r)
# print ("Roots of P2 : ", p2.r)

# # Finding Coefficients
# print ("\n\nCoefficients of P1 : ", p1.c)
# print ("Coefficients of P2 : ", p2.coeffs)

# # Finding Order
# print ("\n\nOrder / Degree of P1 : ", p1.o)
# print ("Order / Degree of P2 : ", p2.order)



# import numpy
# import matplotlib.pyplot as plt

# x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
# y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

# S =numpy.polyfit(x, y, 3) 
# # this line will make coefficients of polynomial equation 
# print("this is the polynomial of coeficeint :",S)
# mymodel = numpy.poly1d(S)
# # this line will fit ploynomial coefficients with the variable and
# # degree according to the polynomial equation 
# print("this is the model \n ",mymodel)
# myline = numpy.linspace(1, 22, 10)
# print("this will predict future values " , mymodel(17))
# print(mymodel(myline))
# plt.scatter(x, y)
# plt.plot(myline, mymodel(myline),'o-r')
# #this line will put the values in x of the polynomial equation 
# plt.show()

#==========================================================================

#The r-squared value ranges from 0 to 1,
#  where 0 means no relationship,
# and 1 means 100% related
# import numpy
# from sklearn.metrics import r2_score
# x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
# y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]
# mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))
# print(r2_score(y, mymodel(x)))
# r squarred value will tell you about the relationship 

#=====================================================================

# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn import linear_model
# df = pd.read_csv("machinelearning.csv")
# print(pd.DataFrame(df))
# df.plot.scatter(x='Weight',y='CO2')
# # df.plot(kind='scatter',x='Weight',y='CO2') # BOTH LINES MEANS THE SAME MEANING
# plt.show()
# X = df[['Weight', 'Volume']]
# y = df['CO2']
# regr = linear_model.LinearRegression()
# regr.fit(X, y)
# na=regr.coef_[0]
# ma=regr.coef_[1]
# print(regr.predict([[790,1000]]))

# #predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm3:
# predictions = regr.predict([[2300, 1300]])
# print(predictions)

#==============================================================================================

# import pandas as pd
# from sklearn import linear_model
# from sklearn.preprocessing import StandardScaler
# df = pd.read_csv("machinelearning.csv")
# # print(pd.DataFrame(df))
# x=df['Weight']
# u=x.mean()
# print(u)
# s=x.std()
# print(s)
# z=(x[0]-u)/s
# mi=df['Weight'].min()
# print(mi)
# ma=df['Weight'].max()
# print(ma)
# zmm=(x[0]- mi) / (ma - mi)# this will show zero because 790 -790 =0
# print("THIS is the value by min and max",zmm)
# print("this is the corelation ", z)

# y=df['Volume']
# m=y.mean()
# std=y.std()
# z=(y[0]-m)/std
# print("this is standard",z)
# #....................................
# import numpy as np
# df = pd.read_csv("machinelearning.csv")
# X = df[['Weight', 'Volume']]
# y=df['CO2']

# scale = StandardScaler()
# scaledX = scale.fit_transform(X)
# print(scaledX)

# regr=linear_model.LinearRegression()
# regr.fit(scaledX,y)
# data=np.array([[2300,1.3]])

# scaled=scale.transform(data)
# print(scaled)

# predicted=regr.predict([scaled[0]])
# print(predicted)

#========================================================================================

# import numpy
# import matplotlib.pyplot as plt
# from sklearn.metrics import r2_score
# numpy.random.seed(2)

# x = numpy.random.normal(3, 1, 100)
# y = numpy.random.normal(150, 40, 100) / x

# train_x = x[:80]
# train_y = y[:80]

# test_x = x[80:]
# test_y = y[80:]

# mymodel = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))

# myline = numpy.linspace(0, 6, 100)

# plt.scatter(train_x, train_y)
# plt.plot(myline, mymodel(myline))
# plt.show()

# r2 = r2_score(test_y, mymodel(test_x))
# print(r2)

# print("this will predict my value", mymodel(5))

#===================================================================================================================
# from pyodide.http import pyfetch
# path= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
# async def download(url, filename):
#     response = await pyfetch(url)
#     if response.status == 200:
#         with open(filename, "wb") as f:
#             f.write(await response.bytes())
# import numpy as np
# msk = np.random.rand(len(df)) < 0.8
# train = cdf[msk]
# test = cdf[~msk]
# from sklearn import linear_model
# regr = linear_model.LinearRegression()
# train_x = np.asanyarray(train[['ENGINESIZE']])
# train_y = np.asanyarray(train[['CO2EMISSIONS']])
# regr.fit(train_x, train_y)
# # The coefficients
# print ('Coefficients: ', regr.coef_)
# print ('Intercept: ',regr.intercept_)
# plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
# plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
# plt.xlabel("Engine size")
# plt.ylabel("Emission")
# from sklearn.metrics import r2_score

# test_x = np.asanyarray(test[['ENGINESIZE']])
# test_y = np.asanyarray(test[['CO2EMISSIONS']])
# test_y_ = regr.predict(test_x)

# print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
# print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
# print("R2-score: %.2f" % r2_score(test_y , test_y_) )

#====================================================================================================


# import requests 
# path='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv'
# async def download(path, filename):
#     # Replace pyfetch with requests.get
#     response = requests.get(path, stream=True)


#     # Check if the response is successful
#     if response.status_code == 200:
#         # Open the file in binary write mode
#         with open(filename, "wb") as f:
#             # Write the response content to the file in chunks
#             for chunk in response.iter_content(chunk_size=1024):
#                 f.write(chunk)
#     else:
#         print("the file is not opened")
#     await download(path,"FuelConsumptionCo2.csv")


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
path="FuelConsumptionCo2.csv"
df=pd.read_csv(path)
mn=pd.DataFrame(df)
print(mn)
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
# cdf.head(9)
# plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
# plt.xlabel("Engine size")
# plt.ylabel("Emission")
# plt.show()

# plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='yellow')
# plt.xlabel("FUELCONSUMPTION_COMB")
# plt.ylabel("Emission")
# plt.show()

# plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS,  color='red')
# plt.xlabel("CYLINDERS")
# plt.ylabel("Emission")
# plt.show()


msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]
from sklearn import linear_model
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (x, y)
# The coefficients
# print ('Coefficients: ', regr.coef_)
# me=cdf['ENGINESIZE']
# print(me.mean())
# mk=cdf['CYLINDERS']
# print(mk.mean())
# mu=cdf['FUELCONSUMPTION_COMB']
# print(mu.mean())
# new_data = np.array([[4, 6,9]])
# predicted_emissions = regr.predict(new_data)
# print('Predicted CO2 emissions:', predicted_emissions)
# filtered_df = df[(df['ENGINESIZE'] >= 4) & (df['ENGINESIZE'] <= 5) &
#                 (df['CYLINDERS'] >= 6) & (df['CYLINDERS'] <= 7) &
#                 (df['FUELCONSUMPTION_COMB'] >= 8) & (df['FUELCONSUMPTION_COMB'] <= 12)]

y_hat = regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
r45=r2_score(test[['CO2EMISSIONS']],y_hat)
print(r45)
# Print the columns of the filtered DataFrame
# print(filtered_df)

