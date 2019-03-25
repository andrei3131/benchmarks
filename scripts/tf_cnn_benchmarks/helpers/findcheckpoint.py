#!/usr/bin/python

import sys
filename = sys.argv[1]
f = open(filename, 'r')
for line in f:
	# DBG> [...]/checkpoints/v-000001/model.ckpt (v. 1)
	if line.startswith("DBG>"):
		if "model.ckpt" in line:
			s = line.split()
			version = s[-1][:-1]
			print("%06d" % int(version))
sys.exit(0)
