import os
# Write your assignment here

my_absolute_dirpath = os.path.abspath(os.path.dirname(__file__))

data_in_1 = open("{0}/data/1.in".format(my_absolute_dirpath), "r")
data_in_2 = open("{0}/data/2.in".format(my_absolute_dirpath), "r")
print('data_in_1')
for line in data_in_1:
    line = line.replace('\n', '')
    print(line)
print()
print('data_in_2')
for line in data_in_2:
    line = line.replace('\n', '')
    print(line)
