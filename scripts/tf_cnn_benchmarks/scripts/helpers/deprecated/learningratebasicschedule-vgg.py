#!/usr/bin/python

import sys

def scheduler(epoch, rate, drop):
	return rate * (0.5 ** (epoch // drop))

# Default values
r = float(sys.argv[1])

drop = 20

s = "%g" % (r)
for e in range(20,250,20):
    # print "%3d: %12.10f" % (e, scheduler(e))
    t = ";%d;%g" % (e, scheduler(e, r, drop))
    s = s + t
print s
