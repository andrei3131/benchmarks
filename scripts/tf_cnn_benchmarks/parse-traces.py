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

class Kernel(object):

    def __init__(self):
        self.order     = 0
        self,name      = ""
        self.duration  = 0.0
        self.occupancy = 0.0


def parse_metrics(file):
    lines = 0
    table = {}
    keywords = ["Stream", "Kernel", "achieved_occupancy"]
    f = open(file, "r")
    for line in f:
        lines += 1
        # Skip first line
        if lines == 1:
            continue
        # Extract attributes of interest
        feature_map['Stream'] = -1

        if lines == 2:
            s = line.split(",")



def parse(file):
    lines = 0
    table = {}
    keywords = ["Stream", "Name", "Duration"]
    f = open(file, "r")
    for line in f:
        lines += 1
        # Skip first line
        if lines == 1:
            continue
