from random import randrange
import numpy as np

def main():
    m = 5
    x = [[randrange(-6, 6) for i in range(m)] +
         [randrange(-11, -6) for i in range(m)] +
         [randrange(6, 11) for i in range(m)],
         [randrange(-6, 6) for i in range(m)] +
         [randrange(-11, -6) for i in range(m)] +
         [randrange(6, 11) for i in range(m)]]
    x = np.transpose(x)
    y = [0] * m + [1] * 2 * m
    print(x)
    print(y)

if __name__ == '__main__':
    main()