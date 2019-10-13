import math
i = 0
n = 100
p = 7

while 1:
    FirstElement = math.floor(i*n/p)
    LastElement = math.floor((i+1)*n/p) - 1
    BlockSize = LastElement - FirstElement
    print("Process:" + i.__str__()
          + " First Element:"
          + FirstElement.__str__()
          + " Last Element:"
          + LastElement.__str__()
          + " Block Size:"
          + BlockSize.__str__())
    i += 1
    if i == 7:
        break
