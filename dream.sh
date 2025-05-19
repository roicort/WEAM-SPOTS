#!/bin/bash

LOGFILE="logs/dream.logs.txt"
n=256

echo "========== RUN RD STARTED: $(date) ==========" > $LOGFILE
echo "" >> $LOGFILE
echo "----- Domain size: $n -----" >> $LOGFILE

# Step -r (Generate images from testing data and associative memories)
START_R=$(date +%s)
uv run python eam.py -r --domain=$n --runpath=runs-$n
END_R=$(date +%s)
DURATION_R=$((END_R - START_R))
echo "    [$(date)] Step -r completed. Time: ${DURATION_R} seconds." >> $LOGFILE

# Step -d (Recurrent generation of memories, 'dreaming')
START_D=$(date +%s)
uv run python eam.py -d --domain=$n --runpath=runs-$n
END_D=$(date +%s)
DURATION_D=$((END_D - START_D))
echo "    [$(date)] Step -d completed. Time: ${DURATION_D} seconds." >> $LOGFILE

echo "    [$(date)] Run for domain $n completed." >> $LOGFILE
echo "------------------------------" >> $LOGFILE

echo "" >> $LOGFILE
echo "========== RUN RD FINISHED: $(date) ==========" >> $LOGFILE
echo "Total run time: $((END_D - START_R)) seconds." >> $LOGFILE