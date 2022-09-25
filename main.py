from allocation import *
from market import *

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    market = Market(50, 100)
    alloc = GreedyUCB(market)

    print(alloc.allocate)
    print(alloc.allocation)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
