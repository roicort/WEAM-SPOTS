#!/bin/bash

for n in 32 64 128 512; do
  # python eam.py -n --domain=$n --runpath=runs-$n &&
  # python eam.py -f --domain=$n --runpath=runs-$n &&
  python eam.py -e 1 --domain=$n --runpath=runs-$n
done
