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
- registers of flag comparison
- Ordinal encoding (one hot but all <=, 1,1,1,0,0,0) 
- normalized features

Reward:
- -1 per step for short solutions
- negative for out of order pairs
- positive for agreement over different tests (collapse space)
- positive all correct
- reward after each step
- negative for duplicate elements
- reward on secret cases
- force overfitting


Other:
- imitation learning
- network layers