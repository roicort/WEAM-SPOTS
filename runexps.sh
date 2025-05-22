#!/bin/bash

runpath=runs-32
echo "Storing results in $runpath"
python eam.py -n --runpath="$runpath" && \
    python eam.py -f --runpath="$runpath" && \
    python eam.py -e 1 --runpath="$runpath" && \
    python eam.py -r --runpath="$runpath" && \
    python eam.py -d --runpath="$runpath"