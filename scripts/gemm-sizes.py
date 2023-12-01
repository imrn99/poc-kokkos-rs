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
        sizes.append(int(size))
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
    plt.semilogx(base=2.0)
    plt.grid(visible=True, axis='y')
    plt.plot(sizes, percentsSlower)
    plt.show()
    

main()