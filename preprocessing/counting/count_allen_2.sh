#!/bin/bash
# generate count matrices 210605

kb count --verbose \
-i ../ref/refdata-gex-mm10-2020-A/kallisto/index.idx \
-g ../ref/refdata-gex-mm10-2020-A/t2g_mm10.txt \
-x 10xv3 \
-o ./allen_B08/ \
-t 30 -m 30G \
-c1 ../ref/refdata-gex-mm10-2020-A/kallisto/cdna_t2c.txt \
-c2 ../ref/refdata-gex-mm10-2020-A/kallisto/intron_t2c.txt \
--workflow lamanno --filter bustools --overwrite --loom \
../datasets/allen_B08/L8TX_190430_01_B08_S01_L003_R1_001.fastq.gz \
../datasets/allen_B08/L8TX_190430_01_B08_S01_L003_R2_001.fastq.gz
