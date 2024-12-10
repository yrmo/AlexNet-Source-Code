import sys
import re

def binary_sum(arr):
    if len(arr) == 1:
        return arr[0]
    mid = len(arr) / 2
    sum_left = binary_sum(arr[:mid])
    sum_right = binary_sum(arr[mid:])
    return [a + b for a,b in zip(sum_left, sum_right)]

def test(path):
    p = re.compile(r'^batch \d+:.*\[((?:[\d\.]+(?:, )?)+)\]}, (\d+)\)\s*$')
    sums = []
    sums2 = []
    ncases = 0
    with open(path) as f:
        for line in f:
            m = p.match(line)
            if m:
                vals = m.group(1).split(',')
                if len(sums) == 0: sums = [0] * len(vals)
                sums = [s + float(v) for s,v in zip(sums, vals)]
                sums2 += [[float(v) for v in vals]]
                ncases += int(m.group(2))
    return [s/ncases for s in sums], [s/ncases for s in binary_sum(sums2)], ncases
    
if __name__ == "__main__":
    errs, errs2, ncases = test(sys.argv[1])
    print errs
    print errs2
    print "--- %d cases" % ncases
