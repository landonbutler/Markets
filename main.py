from allocation import *
from market import *

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    market = Market(5000, 5000)
    alloc = GaleShapley(market)

    alloc.allocate()
    print(alloc.allocation)
