Trade offs:

Instructions:
- Simple: swapGt
- Medium: cmp, cswap
- Complex: cmp, cmov


Observations:
- multiple tests
- all permutations
- with swap registers
- flags
- previous code
- last command

Reward:
- -1 per step for short solutions
- negative for out of order pairs
- positive for agreement over different tests (collapse space)
- positive all correct
- reward after each step
- negative for duplicate elements
- reward on secret cases


Other:
- imitation learning
- network layers