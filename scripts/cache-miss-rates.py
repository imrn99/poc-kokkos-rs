# The script expects a single csv file, containing 3 lines:
# - 1st line: data size (used as the X coordinate)
# - 2nd line: cache miss-rates times using the usual (i.e. naive) layout
# - 3rd line: cache miss-rates times using ideal layout

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
    usualLayoutRates = []
    bestLayoutRates = []
    for size in tmp[0]:
        sizes.append(int(size))
    for time in tmp[1]:
        usualLayoutRates.append(float(time))
    for time in tmp[2]:
        bestLayoutRates.append(float(time))
    tmp.clear()

    # compute relative change
    percentsMore=[]
    for i in range(len(sizes)):
        percentsMore.append( 100 * (usualLayoutRates[i] - bestLayoutRates[i]) / bestLayoutRates[i])

    # plot
    plt.title("GEMM: L1 Cache Miss-Rate Evolution = f(Data Size)")
    plt.xlabel("Square Matrix Dimension (# of rows/cols)")
    plt.ylabel("Miss-Rate (%)")
    
    plt.semilogx(base=2.0)
    plt.grid(visible=True, axis='y')
    plt.scatter(sizes, usualLayoutRates, marker='+', color='r', label="usual-layout")
    plt.scatter(sizes, bestLayoutRates, marker='x', color='b', label="best-layout")
    plt.legend()
    plt.savefig(fname="cache-miss-rates.svg", format="svg")
    

main()
