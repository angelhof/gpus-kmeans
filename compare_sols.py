#!/usr/bin/python
import sys
from numpy.linalg import norm
from numpy import loadtxt


def usage():
    print "### Usage: ./comapre_sols.py output1 output2"


def main():
    if len(sys.argv) < 3:
        print "Incorrect arguments"
        usage()
        return

    out1, out2 = sys.argv[1:]

    t = loadtxt(out1)
    g = loadtxt(out2)

    if t.shape != g.shape:
        print("Result Mismatch")
        return

    t = t - g
    print norm(t)


if __name__ == "__main__":
    main()
