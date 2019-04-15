#!/usr/bin/python

import sys

# Default values
r = float(sys.argv[1])

print "%g;80;%g;120;%g" % (r, r / 10.0, r / 100.)
