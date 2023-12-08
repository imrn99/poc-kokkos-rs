# This script is used to generate a speedup graph from the output
# of the layout-size benchmark (criterion group gemm-sizes).
#
# The script expects a single csv file, containing 3 lines:
# - 1st line: data size (used as the X coordinate)
# - 2nd line: execution times using the usual (i.e. naive) layout
# - 3rd line: execution times using ideal layout

import sys
import csv
import matplotlib.pyplot as plt

def main():
    # read input
    fileName = sys.argv[1]
    tmp = []
    with open(fileName, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            tmp.append(row)

    # parse values
    sizes = []
    usualLayoutTimes = []
    bestLayoutTimes = []
    for size in tmp[0]:
        sizes.append(int(size)) # matrix size = dim1 * dim2 * sizeof(double)
    for time in tmp[1]:
        usualLayoutTimes.append(float(time))
    for time in tmp[2]:
        bestLayoutTimes.append(float(time))
    tmp.clear()

    # compute relative change
    percentsSlower=[]
    for i in range(len(sizes)):
        percentLonger = (usualLayoutTimes[i] - bestLayoutTimes[i]) / bestLayoutTimes[i]
        percentsSlower.append(- 100*100 * percentLonger / (100.0 + percentLonger))
    
    # plot
    plt.title("GEMM: Speed Gain = f(Data Size)")
    plt.xlabel("Square Matrix Dimension (# of rows/cols)")
    plt.ylabel("Gain (%)")
    plt.ylim([-175, 10])
    plt.semilogx(base=2.0)
    plt.axvline(x=64*6**0.5,  label="Exceed L1 Total Size", color='r', ymax=0.95, ymin=0.05)
    plt.axvline(x=512*3**0.5, label="Exceed L2 Total Size", color='g', ymax=0.95, ymin=0.05)
    plt.axvline(x=2048,       label="Exceed L3 Total Size", color='b', ymax=0.95, ymin=0.15)
    plt.legend(loc="center left")
    plt.grid(visible=True, axis='y')
    plt.scatter(sizes, percentsSlower, marker='+', color='r')
    plt.savefig(fname="gemm-sizes-plot.svg", format="svg")
    

main()
