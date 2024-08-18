#!/bin/bash

# Loop 5 times
for i in {1..10}
do
   echo " === Executing iteration $i ==="
   python GW170817_seed.py
done

echo "All iterations completed!"