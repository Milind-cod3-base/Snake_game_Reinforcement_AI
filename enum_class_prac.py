""" Dummy module which contains Enum class for practice"""

# from enum module imported Enum class 
from enum import Enum

# making a class Days inheriting from class ENUM from module
class Days(Enum):
    Sun =1
    Mon = 2
    Tue =3 


print('enum memeber accessed by name: ')
print(Days['Mon'])

print('enum member accessed by Value: ')
print(Days(1))