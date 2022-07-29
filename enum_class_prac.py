""" Dummy module which contains Enum class for practice"""

# from enum module imported Enum class 
from enum import Enum

# making a class Days inheriting from class ENUM from module
class Days(Enum):
    Sun =1
    Mon = 2
    Tue =3 


# hashing the enum members (giving keys)
# setting them into a dictionary
Daytype = {} # empty dicitonary created

Daytype[Days.Sun] = 'Sun God'    #key-value pair generated
Daytype[Days.Mon] = "Moon God"