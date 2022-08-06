from itertools import islice

print(islice('ABCDEFG', 2))
a = 'asdad_dadas.jpg'
print(islice(a, 4))
print(a.split('_')[1][-3:])