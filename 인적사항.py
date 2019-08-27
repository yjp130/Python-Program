personal_information={
    "name" : "윤종필",
    "gender" : "남자",
    "email" : "yjp130@gmail.com",
    "age" : "29"
}

print('personal_information.keys() \t= ',personal_information.keys())
print('personal_information.values() \t= ',personal_information.values())
print('personal_information.items() \t= ',personal_information.items())
print('-'*130)

for key, value in personal_information.items():
    print('%s \t : %s' %(key, value))


