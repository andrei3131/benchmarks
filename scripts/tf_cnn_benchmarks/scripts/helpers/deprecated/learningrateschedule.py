#!/usr/bin/python

import sys

# Default values
r0=0.1
b0=128
g0=1

b = int(sys.argv[1])
g = int(sys.argv[2])

r = r0
# Effective batch size during training
b = b * g
if b > b0:
   scale = int(b / b0)
   r = r0 * float(scale)
print "%g;80;%g;120;%g" % (r, r / 10.0, r / 100.)
