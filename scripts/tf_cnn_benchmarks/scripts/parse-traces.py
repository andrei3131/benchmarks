# One file starts with:
#
# Line 1: "======== Profiling result:"
#      2: Attribute names
#      3: Attribute types
#
# An example:
#
# Device            : "TITAN X (Pascal) (0)",
# Context           : "3",
# Stream            : "78",
# Kernel            : "void function(args),
# Correlation ID    : 2298,
# Achieved occupancy: 0.664158
#
# The other file starts with:
#
# Line 1: "======== Profiling result:"
#      2: Attribute names
#      3: Attribute types
#
# An example:
#
# Start (s)          : 0.944832,
# Duration (us)      : 10.081000,
# Grid X             : 56,
# Grid Y             : 1,
# Grid Z             : 1,
# Block X            : 1024,
# Block Y            : 1,
# Block Z            : 1,
# Registers/thread   : 30,
# Static memory (KB) : 0.000000,
# Dynamic memory (KB): 0.000000,
# Size (KB)          : ,
# Throughput (GB/s)  : ,
# Source memory type : ,
# Dest. memory type  : ,
# Device             : "TITAN X (Pascal) (0)",
# Context            : "4",
# Stream             : "78",
# Name               : "void function (args)"
# Correlation ID     : 2384
#

import sys
import csv


class Kernel(object):

    def __init__(self):
        self.order = 0
        self.name = None
        self.stream = 0
        self.duration = 0.0
        self.occupancy = 0.0
        self.matched = False


def parse(filename, metrics=False):
    kernels = []
    lines = 0
    order = 1
    table = {}
    keywords = []
    if metrics:
        keywords = ["Stream", "Kernel", "achieved_occupancy"]
    else:
        keywords = ["Stream", "Name", "Duration"]
    f = open(filename)
    reader = csv.reader(f, delimiter=',', quotechar='"')
    for s in reader:
        lines += 1
        # Skip first line
        if lines == 1:
            continue
        # Second line lists attribute names
        if lines == 2:
            # Extract attributes of interest
            index = 0
            for attribute in s:
                if attribute in keywords:
                    # print("Found %s at position %d" % (attribute, index))
                    table[attribute] = index
                index += 1
            continue
        # Third line lists attribute types
        if lines == 3:
            continue
        # Parse results
        k = Kernel()
        k.order = order
        order += 1
        if metrics:
            k.name = s[table['Kernel']].strip()
            k.stream = int(s[table['Stream']].strip('\"'))
            k.occupancy = float(s[table['achieved_occupancy']])
            k.duration = 0.0
        else:
            k.name = s[table["Name"]].strip()
            k.stream = int(s[table['Stream']].strip('\"'))
            k.occupancy = 0.0
            k.duration = float(s[table["Duration"]])
        kernels.append(k)
    print("%d kernels in %s" % (len(kernels), filename))
    return kernels


def find(kernel, kernels):
    found = False
    value = 0
    stream = 0
    for k in kernels:
        if kernel.name == k.name and k.matched == False:
            found = True
            value = k.order
            stream = k.stream
            k.matched = True
            break
    return found, value, stream


if __name__ == '__main__':
    K = parse(sys.argv[1], metrics=True)
    L = parse(sys.argv[2])

    Kmap = {}
    for k in K:
        if k.name not in Kmap:
            Kmap[k.name] = 1
        else:
            Kmap[k.name] += 1

    Lmap = {}
    for l in L:
        if l.name not in Lmap:
            Lmap[l.name] = 1
        else:
            Lmap[l.name] += 1

    print(len(Kmap))
    print(len(Lmap))
    for k in Kmap:
        if k in Lmap:
            print("%d %d" % (Kmap[k], Lmap[k]))
            if Kmap[k] > Lmap[k]:
                print k
        else:
            print("error: %d %d" % (Lmap[k], 0))
            sys.exit(1)

    # match = 0
    # for a in K:
    #     found, order, stream = find(a, L)
    #     if not found:
    #         print("error: kernel not found: %s (%2d.%4d))" % (a.name, a.stream, a.order))
    #         sys.exit(1)
    #     print("Current order %2d.%4d %2d.%4d" % (a.stream, a.order, stream, order))
    #     match += 1
    # print("%d kernels match" % match)
    sys.exit(0)
