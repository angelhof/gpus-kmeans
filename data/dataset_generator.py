#!/usr/bin/python

import sys
from random import randint, uniform

MINVAL = -pow(2, 10)
MAXVAL = +pow(2, 10)


def usage():
    print "---Usage: ./dataset_generator n k dim mode"
    print "mode 0 : Generates int"
    print "mode 1 : Generates floats"


def main():
    print "---K-Means Dataset Generator---"
    usage()

    if len(sys.argv) < 5:
        print "Incorrect arguments"
        usage()
        return

    # Parse arguments
    n, k, dim, mode = map(int, sys.argv[1:])

    f = open("_".join(["dataset", str(n), str(k), str(dim), str(mode)]), 'w')
    # Write descriptors
    f.writelines(" ".join(map(str, [n, k])) + '\n')

    for i in range(n):
        if mode == 0:
            line = " ".join(map(str, [randint(MINVAL, MAXVAL)
                                      for i in range(dim)])) + "\n"
        elif mode == 1:
            line = " ".join(map(str, [uniform(MINVAL, MAXVAL)
                                      for i in range(dim)])) + "\n"
        f.writelines(line)

    f.close()


if __name__ == "__main__":
    main()
