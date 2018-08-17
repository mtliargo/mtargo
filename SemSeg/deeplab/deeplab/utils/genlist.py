n = 60000

with open(r'D:\Data\CarlaGen\C20_S1\train-0.1.txt', 'w') as f:
  for i in range(0, n, 10):
    f.write('%08d.png\n' % (i + 1))

