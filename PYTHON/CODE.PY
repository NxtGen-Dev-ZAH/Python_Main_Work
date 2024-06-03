# A=5
# B=89
# SUM=A+B
# print(SUM)
# b : str = 'line 1'\
# ' line 2'\
# ' line3'

# print(b)
# y =int(3) # this is explicit casting
# x=str("saeed")
# z=float(78.33)
# print(type(x))
# print(type(y))#this will print type
# camel case => myVariableName
# pascal case=> MyVariableName
# snake case => my_variable_name
# j=k=l="orange"
# e,f,g="mango","pineapple","banana"
# i =0
# while  i<5 :
#     age =int (input("enter the age of the student of five students : "));
#     if(age>18):
#         age=age +5
#     if (age<20):
#         age = age -7
#     i+=1
#     print ("the age you entered is ",age);
"""word = "hello pakistan"
for lettersintheword in word :
    print(lettersintheword)
fruits=("banana","tomato","grape","pomegranade")
for fruit in fruits:
    print(fruit)"""
"""
m =0
while m<5 :
    print ("zaheer is greatest of all time")
    if (m==3) :
        m=m+2
    m+=1
"""
"""
print ("i am pakistani")
a,b,c = 6,5.6,"apple"
print(a,b,c)
# import CONSTANTS
# print(CONSTANTS.PI)
# print("the value of gravity near the surface of earth is",CONSTANTS.GRAVITY)
# print(CONSTANTS.MYNAMEIS)

fruits = ["apple", "mango", "orange"]
for fruit in fruits :
    print(fruits)

# tuple literal
numbers = (1, 2, 3)
j =1
while j<5 :
    print(numbers)
    j+=1
# dictionary literal
alphabets = {"a":67, "b":78, "c":99}
print(alphabets["a"])
alphabets["d"]=456
for alphabet in alphabets:
    print(alphabets[alphabet])
if "b" in alphabets  :
    print("yes")
print(alphabets.keys())
print(alphabets.values())
i=7
if (i==6) :
# set literal
    vowels = {'a', 'e', 'i' , 'o', 'u'}
    print(vowels)
elif (i==5) :
    print("i am the danger")

"""
# x = ["apple", "banana", "cherry",84,8.4,[8,9]]#-> list
# print(x[2])
"""
j={"banana","apple","cherry",34}#set
k=("asdf","jkl;","zxcv","bnm,","qwert","yuiop")#tuple
l={"name":"zaheer","age":19,"work experience":" 2 years"}#dictionary
#explicit casting
x = list (("apple", "banana", "cherry",84,8.4,[8,9]))#-> list
j=set(("banana","apple","cherry",34))#set
k=tuple(("asdf","jkl;","zxcv","bnm,","qwert","yuiop"))#tuple

# dictionary= 
l=dict(name="zaheer",age=19,workexperience= "2 years")#dictionary
for a in l :
  print(l[a])

 

#bool,str,int,float,list,tuple,set,dict

print(type(j))
print(type(k))
print(l)
"""
# c=5j+7#j is reffered as imajinary part
# print(type(c))
# print(c)
# you can use 3 double quotes to indicate a variable
# a = """Lorem ipsum dolor sit amet,
# consectetur adipiscing elit,
# sed do eiusmod tempor incididunt
# ut labore et dolore magna aliqua."""
# print(a)
"""
a = "Hello, World!"
print(len(a))
# for ax in a:
#     print(ax)
if "He" in a :#to check we use in
    print ("yes found")# to check if the desired thing is present or not
print(a[2:5]) # the last index is not included
print(a[::3])# Slice From the Start means it will print frist and will skip next 2 and will print third
#print(a[5:]) Slice to the end
#var.upper()
#var.lower()
#var.replace()
#split() splits the string into substrings
m=str('in the name of god')
print(m.split("o"));
a="i am zaheer "
b= "ahmed {}"
d=a+b
c =19
print(d.format(c))#value inside the format function
#shall be numeric + there should a {} at the end of the string
print("{}{}".format(a,c))#.format()must be used with curly brackets"""
# no = 3
# itemno = 567
# price = 49.95
# myorder = "I want to pay {1} dollars for {2} pieces of item {0}."
# print(myorder.format(itemno, price,no))

thislist = ["apple", "banana", "cherry", "orange", "kiwi", "melon", "mango"]
# print(thislist[-4::])
# # if "apple" in thislist :
#     print("apple is found")
# thislist[1]="peach"#this is how replacing works
# print(thislist);
# thislist[1:3]=["blackberry","watermelon"]# IT WILL REPLACE FROM INDEX 1-3
# thislist.insert(2,"strawberry")
# print(thislist);
# print(len(thislist))
# # .append() add an item to the end of the list
# #.extend () we can append elements from another list to current list
# thislist.remove("strawberry")
# print(thislist)
# thislist.pop(3)# THIRD INDEX ELEMENT GETS POPPED
# del thislist[2]
# print(thislist)
# del thislist ;thislist.clear ()

# for i in range(4):
#     print(thislist[i])


# fruits = ["apple", "banana", "cherry", "kiwi", "mango"]
# newlist = []
# i=0
# for x in fruits:
#     if "a" in x :
#         newlist.insert(i,x)
#         i=i+1
# print(newlist)
# variable.sort() sorts the value,CASE SENSITIVE CAPITAL LETTER FIRST
# .sort(reverse=TRUE) FOR DESENCENDING SORT
# ABS()MEANS ABSOLUTE DIFFERENCE MEANS ALL POSITIVE
# accessing tuple elements using slicing
# my_tuple = ('p', 'r', 'o', 'g', 'r', 'a', 'm', 'i', 'z')

# print(my_tuple[1:3])

# # elements beginning to 2nd
# print(my_tuple[:-7]) # prints ('p', 'r')

# print(my_tuple[7:]) # prints ('i', 'z')

# x = ("apple", "banana", "cherry")
# y = list(x)
# y[1] = "kiwi"
# x = tuple(y)
# z=list(x)
# z[2]="mango"
# z=tuple(z)
# print(z)
# print(x)

# True and 1 is considered the same value so it will be ignored
# thisset = {"apple", "banana", "cherry", True, 1, 2}
# print(thisset)
# #.add() is used to a specific item in the data items
# #.update() is used to add data item to an existing data items
# # The union() method returns a new set with all items from both sets
# #The intersection() method will return a new set, that only contains the items that are present in both sets.
# model={"1567":"1980"}
# thisdict = {
#   "brand": {"Ford":"shellby","ferrari":"thomas"},
#  "model":model,
#  "year": 1964
# }
# print(thisdict["brand"])
# print(thisdict["model"]["1567"])
# print(thisdict.get("year"))

# salary = 15000
# balance = 8000
# if salary ==balance  :
#     print("its the start of month")
# elif  balance <= 0.5*salary:
#     print("this is the end of month")
# else :
#     print("its the mid of month");

# A=900
# B=900
# print(A+B) if A>B else print(B-A), print(A*B)if A==B else print("-");

# a = 33
# b = 200
# if not a > b:
#   print("a is NOT greater than b")

# import random as rd

# targetno = rd.randint(1,100)

# attempts=0
# while True :
#     try :
#         guess=int(input("enter your guess no "))
#     except ValueError:
#         print("you have entered an incorrect no ")
#         continue
#     attempts+=1

#     if guess == targetno :
#         print("congratulations you have succedded")
#         break
#     elif guess>targetno:
#         print("your entered number is higher")
#     else :
#         print("your entered number is Less")

#     play_again = input("Do you want to play again? (yes/no): ")
#     if play_again.lower() != "yes":
#         break

# print ("thanks for playing")
# print("you have taken total attempts " ,attempts);

# balance=1000
# print("Welcome to the ATM Simulator!")
# print("Available Options:\n","1. Check Balance\n","2.MAKE A WITHDRAWL\n","QUIT\n")

# while True:
#     try:
#         user_select= int(input("Enter your choice (1/2/3):"))
#     except ValueError :
#         print("you entered incorrect option")
#         continue

#     if user_select==1:
#         print("Your current balance is $",balance)
#     elif user_select==2:
#         withdraw=int(input("Enter the withdrawal amount:"))
#         if withdraw >balance:
#             print("Error: Insufficient funds. Your current balance is",balance)
#         else:
#             balance=balance-withdraw
#             print("Withdrawal successful. Your new balance is",balance)
#     elif user_select==3:
#         print("GOODBYE !")

# for x in range(2,30,3):
#   if x == 23 or x==  24 :
#     continue
#   print(x)

# adj = ["red", "big", "tasty"]
# fruits = ["apple", "banana", "cherry"]
# for x in adj:
#   for y in fruits:
#     print(x, y)

# M=int(78)#this is global variable
# def func():
#     i=99#this is local variable
#     print(i+M)
# func()

# y="i am the only best person in the town"
# def function():
#     global y
#     y="awesome"
#     print("my tareef is",y)
# function()
# print(y)

# def intro(fname,lname) :
#     print("the person full intro is ",fname+" "+lname)

# intro("zaheer","ahmed")

# #tuple of arguments
# def myfunction(*colors):
#     print("the favourite color is ",colors[2])

# myfunction("black","Grey","White")

# #dictionary of arguments
# def my_function(**kid):
#   print("His last name is " + kid["lname"])

# my_function(fname = "Tobias", lname = "Refsnes")


# def my_function(country = "Norway"):
#   print("I am from " + country)

# my_function("Sweden")
# my_function("India")
# my_function()
# my_function("Brazil")

# x = lambda a, b, c : a + b + c
# print(x(5, 6, 2))

# def myfunc(n):
#   return lambda a : a * n

# mydoubler = myfunc(2)
# mytripler = myfunc(3)

# print(mydoubler(11))
# print(mytripler(11))

# =================================================================

# class person:
#   def __init__(self,name,age,gender) :
#     self.name=name
#     self.age=age
#     self.gender=gender

#   def __str__(self) :
#     return f"{self.name} {self.gender}"
#   def myfunctions(self) :
#     print(self.name,self.age,self.gender)

# # p1=person("MOIZ","20","male")
# # p1.myfunctions()
# # print(p1.name)
# # print(p1.gender)
# # print(p1)

# # # p1.age = 40
# # del p1.age
# # del p1

# class student(person):
#   def __init__(self, name,age,gender,classroom,subjects):
#     person.__init__(self,name, age, gender)
#     self.classroom=classroom
#     self.subjects=subjects

#   def __str__(self):
#     return f"{self.name} {self.gender} {self.classroom}"

# student1=student("USAMA",23,"MALE",12,8)
# print(student1)

# class CLASSCR(person):
#   def __init__(self, name, age, gender,total_marks,obtained_marks):
#     super().__init__(name, age, gender)
#     self.total_marks=total_marks
#     self.obtained_marks=obtained_marks

# student=CLASSCR("AREEJ",19,"FEMALE",360,220)

# class ANIMAL :
#   def __init__(self ,name , species):
#     self.name=name
#     self.species= species
#     def speak(self):
#      pass

# class lion(ANIMAL):
#   def speak(self):
#     return "roar"

# class elephant (ANIMAL):
#   def speak(self):
#     return "CHHEEEN"

# class monkey (ANIMAL):
#   def speak(self):
#     return "djalj"

# class zoo:
#   def __init__(self):
#     self.animals=[]
#   def addanimals(self,animals):
#     self.animals.append(animals)
#   def listanimals(self):
#     for x in self.animals:
#       print(f"{x.name} ({x.species}) : {x.speak()}")
#   def makeanimalsspeak(self):
#     for anim in self.animals:
#       print(f"{anim.name} says {anim.speak()}")

# lion1 = lion("Simba", "Lion")
# elephant1 = elephant("Dumbo", "Elephant")
# monkey1 = monkey("George", "Monkey")

# zoo1 = zoo()

# zoo1.addanimals(lion1)
# zoo1.addanimals(elephant1)
# zoo1.addanimals(monkey1)
# zoo1.listanimals()
# zoo1.makeanimalsspeak()


# class author:
#   def __init__ (self,name,birthyear ):
#     self.name=name
#     self.birthyear=birthyear
#     self.books=[]

# class book:
#   def __init__(self,title,publicationyear,author):
#     self.title=title
#     self.publicationyear=publicationyear
#     self.author=author

# class patron :
#   def __init__(self,name):
#     self.name =name
#     self.borrowed_books=[]
#   def borrowedbooks(self,book):
#     self.borrowed_books.append(book)
#   def return_book(self, book):
#     self.borrowed_books.remove(book)

# class library:
#   def __init__(self):
#     self.libcatalog=[]
#     self.patrons=[]

#   def addbooks(self,book):
#     self.libcatalog.append(book)

#   def addpatrons(self,patron):
#     self.patrons.append(patron)

#   def borrowedbooks(self,patron,book):
#     if book in self.libcatalog :
#       self.libcatalog.remove(book)
#       patron.borrowedbooks(book)
#       print(f"{patron.name} has borrowed '{book.title}")
#     else:
#       print("book not available in the library")

#   def return_book(self, patron, book):
#     if book in patron.borrowed_books:
#       patron.return_book(book)
#       self.libcatalog.append(book)
#       print(f"{patron.name} has returned '{book.title}'.")

#   def list_books_borrowed(self, patron):
#     print(f"Books borrowed by {patron.name}:")
#     for book in patron.borrowed_books:
#       print(f"'{book.title}' by {book.author.name} ({book.publicationyear})")

# author1 = author("J.K. Rowling", 1965)
# author2 = author("George Orwell", 1903)

# # Create book objects
# book1 = book("Harry Potter and the Sorcerer's Stone", 1997, author1)
# book2 = book("1984", 1949, author2)

# # Create patron objects
# patron1 = patron("Alice")
# patron2 = patron("Bob")

# # Create a library
# library = library()

# # Add books to the library
# library.addbooks(book1)
# library.addbooks(book2)

# # Add patrons to the library
# library.addpatrons(patron1)
# library.addpatrons(patron2)

# # Allow patrons to borrow books
# library.borrowedbooks(patron1, book1)
# library.borrowedbooks(patron2, book2)

# # List books borrowed by each patron
# library.list_books_borrowed(patron1)
# library.list_books_borrowed(patron2)

# # Allow patrons to return books
# library.return_book(patron1, book1)

# # List books borrowed by each patron again
# library.list_books_borrowed(patron1)
# library.list_books_borrowed(patron2)

# GLOBAL KEYWORD CAN CHANGE THE SCOPE OF ANY VARIABLE TO GLOBAL

# ========================================================
# import MYMODULE as MX

# MX.GREETINGS("ZAHEER")
# MX.sum(8,9)

# print(MX.person1["age"])

# ========================================================

# min(),max( , , ) , abs( ), pow(a,b), import math ,math.sqrt( )
# math.ceil( )Rounds x up to the nearest integer.,
#  math.floor( )Rounds x down to the nearest integer, math.pi

# ========================================================
# try:
#    print("my name is zaheer",x)
# except:
#    print("something went wrong in the code ......")
# else :
#    print("there is the error in the code ")
# ========================================================
# ID = int (input("ENTER THE ID FOR THE PASSCODE TO DETECT ::  "))
# print(ID)
# age = 36
# name = "John"
# txt = "His name is {1}. {1} is {0} years old."
# print(txt.format( age , name ))

# =========================================================
# f = open("demofile.txt","r")
# print(f.readline(5))
# readline reads the whole line
# read(5) first five character
# f.close()
# f=open("demofile.txt","a")#a means append the file
# f.write("\n now this file is in appended and the data is being written in it")
# f.close()

# f = open("demofile.txt","r")
# print(f.read())
# f.close()

# f = open("demofile.txt","w")#w means in write mode
# f.write("whooops the data is being deleted")
# f.close()

# f = open("demofile.txt","r")
# print(f.read())
# f.close()


# import os
# os.remove("newfile.txt")

# k=open("newfile.txt","x")#x is used to create a file
# k.write("i am handsome young lad")
# k.close()
# k=open("newfile.txt","a")
# k.write("I AM WORKING HARD DOING MY DAY NIGHT ONE TO MEET MY EXPENSES MEET")
# k.close()
# k=open("newfile.txt","r")
# print(k.read())
# k.close()
# #if os.path.exists
# #os.rmdir() remove a directory,folder

# a="hello mike".find("mike")
# print('1'+'2')


def ABCD(y):
    print(ratings)
    return ratings + y


ratings = 9

z = ABCD(2)
print(z)
print(ratings)

L = [23, 45, 67, 11]
m = sorted(L)
print(m)
L.sort()
print(L)
# if a variable is not defined in function local
# scope then the variable will be assigned a value
# from the global(keyword) scope with the variable of same name


import html5lib
from matplotlib import artist, table
import requests

url = "http://www.ibm.com/"
r = requests.get(url)
print(r.status_code)
# print(r.request.headers)
header = r.headers
print(header)
print(header["date"])
print(r.encoding)
print(r.text[0:50])

import requests

url = "http://httpbin.org/get"
payload = {"NAME": ["Zaheer", "USAMA", "ZOHAIB"], "ID": ["1234", "2345", "5678"]}
R = requests.get(url, params=payload)
print(R.url)
print(R.text)
print(R.headers["Content-Type"])
url_post = "http://httpbin.org/post"
payload = {"name": ["ABBSASI", "KHAN", "MALIK"], "ID": ["1234", "5678", "9087"]}
r_request = requests.post(url_post, payload)
print("THIS IS THE POST", r_request.url)
print("this is the body", r_request.request.body)


from bs4 import BeautifulSoup

html = "<!DOCTYPE html><html><head><title>Page Title</title></head><body><h3><b id='boldest'>Lebron James</b></h3><p> Salary: $ 92,000,000 </p><h3> Stephen Curry</h3><p> Salary: $85,000,000 </p><h3> Kevin Durant </h3><p> Salary: $73,200,000</p></body></html>"

soup = BeautifulSoup(html, "html5lib")
ta = soup.title
tab = soup.h3
print(ta)
print(soup.h3)

tag_child = tab.find("b")
print(tag_child.text)

parent = tag_child.parent
print(parent)

sib = tab.next_sibling
print(sib.text)
print(tag_child.attrs)


from bs4 import BeautifulSoup

html2 = "<table><tr><td>Pizza Place</td><td>Orders</td><td>Slices</td></tr><tr><td>Domino's Pizza</td><td>10</td><td>100</td></tr><tr><td>Little Caesars</td><td>12</td><td>144</td></table>"
soup2 = BeautifulSoup(html2, "html5lib")
tables = soup2.find("table")
table_rows = tables.find_all("tr")
print(table_rows)
table3 = table_rows[2]
print(table3)

for i, row in enumerate(table_rows):
    print("row", i)
    cells = row.find_all("td")
    # here i is index and row will contain value or data
    for j, cell in enumerate(cells):
        print("column", j, "cell", cell.text)

import requests
from bs4 import BeautifulSoup

page = requests.get("http://www.ibm.com").text
soup = BeautifulSoup(page, "html5lib")

artists = soup.find_all("a")
for artist in artists:
    names = artist.contents[0]
    fulllink = artist.get("href")
    print(names)
    print(fulllink)

# table_bs.find_all(id="flight") table_bs=>soup
# table_bs.find_all(href=True)
# table_bs.find_all(string="Florida")
# tables = pd.read_html(url)

var = "01234567"
print(var[::2])
name = "lizz"
print(name[0:2])
print(1 / 2)
print(float(3))
x = "2"
if x != 1:
    print("thsi is me")
else:
    print("it is not me")

A = ["1", "2", "3"]
for a in A:
    print(2 * a)
