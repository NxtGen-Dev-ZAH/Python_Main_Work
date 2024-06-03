a="usman"
print (a)

a:int=7
print(a)
print(dir((a)))
print(type(a))
a:list[str]=["zaheer is working","this is python",[34,56,"this is data"],"dushman ke code"]
for x in a :
  print(x)
my_list = [1, 2, 3, 4, 5] 
my_list.insert(2, 6) 
print(my_list)

a:tuple[int]=(4,5,3,1)
for x in a :
  print(x)
print(a[0:3]);
TUPELI=sorted(a)
print(TUPELI)
a2=a+( "world class",("data coultd be anything","this data"),("this data is ","that data"),"extreme data ")
print(a2)
print(a2[5][1])


b={'a':9,'b':7,'d':6,'c':67}
for x in b :
  print(b[x])

b=str("this is python ")
print(b)
print(type(b))


for i,x in enumerate(['A','B','C']) :
  print(i+1,x)


class points():
  def __init__(self,x,y):
   self.x=x
   self.y=y

  def printpo(self):
    print('x =' ,self.x,'y =',self.y)
  
p3=points(4,6)
p3.x=7
p3.printpo()

def add(a,b):
   return a+b;
a=add(3,5)
print(a)


with open("workinfile.txt","r") as file2:
  filemm=file2.read()
  print(filemm)
print(file2.closed)

file1=open("workinfile.txt","r")
for line in file1:
  print(line)

filem=open("workinfile.txt","r")
# print(filem.read(),"\n this is the next line \n")
line1=filem.readline()
line2=filem.readline()
filem.close()
#readline for loop
print(line2)
filem=open("workinfile.txt","a")
filem.write("\n  this is data on the next line ")
print(filem.name)

if 'primary' in line2 :
    print('This line is important!')

#file.seek(10)  # Move to the 11th byte (0-based index)
 

