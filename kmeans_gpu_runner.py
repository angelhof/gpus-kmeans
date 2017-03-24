#!/usr/bin/python
import sys
import subprocess

infile = open("./data/input.in")


def usage():
    print "### Usage: ./kmeans_gpu_runner.py BLOCKSIZE_MIN BLOCKSIZE_MAX STEP ITERATIONS"


def list_average(l):
    av = sum(l) / len(l)
    return round(av, 6)


def main():
    print "### Runner ###"
    usage()
    print "##############"

    if len(sys.argv) < 5:
        print "Incorrect arguments"
        usage()
        return

    # Parse arguments
    blockmin, blockmax, step, iters = map(int, sys.argv[1:])

    outputs = []
    for i in range(blockmin, blockmax + step, step):
        print "Executing with ", i, "blocks"
        # try:
        current_times = []

        for j in range(iters):
            try:
                p = subprocess.Popen(
                    ["./GPU/kmeans_gpu", str(i)], stdin=infile)
                p.wait()
            except IOError:
                print("Executable not found")
                return
            infile.seek(0)  # Restore Cursor
            # Get program output with accurate time readings
            f = open("log.out")
            current_times.append(float(f.readline().split(":")[1]))
            f.close()

        outputs.append((i, list_average(current_times)))

    print 10 * "#"
    print "Experiment Results"
    print "Each case was run " + str(iters) + " times"

    for o in outputs:
        print o

if __name__ == "__main__":
    main()
