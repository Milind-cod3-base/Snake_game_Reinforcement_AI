""" Dummy module which contains Enum class for practice"""

# from enum module imported Enum class 
from enum import Enum

# making a class Days inheriting from class ENUM from module
class Days(Enum):
    Sun =1
    Mon = 2
    Tue =3 


# print the enum member as a string
print("The enum memeber as a string is: ", end="")
print(Days.Mon)
