import collections


gap = collections.OrderedDict()

name = ['dh', 'yzx', 'lcx', 'hlh', 'gyy']

for i in name:
    h, m = [int(x) for x in input().split(" ")]
    gap[i] = m - h
c = collections.Counter(gap).most_common()  # 返回一个列表，按照dict的value从大到小排序

for i, j in c:
    print(i, j)