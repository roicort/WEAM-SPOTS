#!/bin/bash

LOGFILE="logs/train.logs.txt"

echo "========== TRAINING STARTED: $(date) ==========" > $LOGFILE

for n in 256; do
    echo "" >> $LOGFILE
    echo "----- Domain size: $n -----" >> $LOGFILE

    # Step -n
    START_N=$(date +%s)
    uv run python eam.py -n --domain=$n --runpath=runs-$n
    END_N=$(date +%s)
    DURATION_N=$((END_N - START_N))
    echo "    [$(date)] Step -n completed. Time: ${DURATION_N} seconds." >> $LOGFILE

    # Step -f
    START_F=$(date +%s)
    uv run python eam.py -f --domain=$n --runpath=runs-$n
    END_F=$(date +%s)
    DURATION_F=$((END_F - START_F))
    echo "    [$(date)] Step -f completed. Time: ${DURATION_F} seconds." >> $LOGFILE

    # Step -e
    START_E=$(date +%s)
    uv run python eam.py -e 1 --domain=$n --runpath=runs-$n
    END_E=$(date +%s)
    DURATION_E=$((END_E - START_E))
    echo "    [$(date)] Step -e completed. Time: ${DURATION_E} seconds." >> $LOGFILE

    echo "    [$(date)] Training for domain $n completed." >> $LOGFILE
    echo "------------------------------" >> $LOGFILE
done

echo "" >> $LOGFILE
echo "========== TRAINING FINISHED: $(date) ==========" >> $LOGFILE
echo "Total training time: $((END_E - START_N)) seconds." >> $LOGFILE