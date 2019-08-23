import random

group=[]
newgroup=[]
studentnum=int(input("선생님, 조교, 학생 수의 합을 입력 하시오:    "))


for i in range(1,studentnum+1,1):
    num=random.randint(1,studentnum)
    
    while num in group:
        num=random.randint(1,studentnum)

    group.append(num)
    newgroup.append(num)
    if i%4==0:
        print(newgroup)
        newgroup=[]
print(newgroup)

        
