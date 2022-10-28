#!/bin/bash
# generate count matrices 220601
# weve quantified four of the allen samples, its time to do the other eight

#kb count --verbose \
#-i ../ref/refdata-gex-mm10-2020-A/kallisto/index.idx \
#-g ../ref/refdata-gex-mm10-2020-A/t2g_mm10.txt \
#-x 10xv3 \
#-o ./allen_A01/ \
#-t 50 -m 50G \
#-c1 ../ref/refdata-gex-mm10-2020-A/kallisto/cdna_t2c.txt \
#-c2 ../ref/refdata-gex-mm10-2020-A/kallisto/intron_t2c.txt \
#--workflow lamanno --filter bustools --overwrite --loom \
#../datasets/allen_A01/L8TX_181211_01_A01_S01_L003_R1_001.fastq.gz \
#../datasets/allen_A01/L8TX_181211_01_A01_S01_L003_R2_001.fastq.gz

#source ~/.bashrc

echo "Starting D01..."

kb count --verbose \
-i ../ref/refdata-gex-mm10-2020-A/kallisto/index.idx \
-g ../ref/refdata-gex-mm10-2020-A/t2g_mm10.txt \
-x 10xv3 \
-o ./allen_D01/ \
-t 20 -m 50G \
-c1 ../ref/refdata-gex-mm10-2020-A/kallisto/cdna_t2c.txt \
-c2 ../ref/refdata-gex-mm10-2020-A/kallisto/intron_t2c.txt \
--workflow lamanno --filter bustools --overwrite --loom \
../datasets/allen_D01/L8TX_181211_01_D01_S01_L003_R1_001.fastq.gz \
../datasets/allen_D01/L8TX_181211_01_D01_S01_L003_R2_001.fastq.gz \

kb count --verbose \
-i ../ref/refdata-gex-mm10-2020-A/kallisto/index.idx \
-g ../ref/refdata-gex-mm10-2020-A/t2g_mm10.txt \
-x 10xv3 \
-o ./allen_E01/ \
-t 50 -m 50G \
-c1 ../ref/refdata-gex-mm10-2020-A/kallisto/cdna_t2c.txt \
-c2 ../ref/refdata-gex-mm10-2020-A/kallisto/intron_t2c.txt \
--workflow lamanno --filter bustools --overwrite --loom \
../datasets/allen_E01/L8TX_181211_01_E01_S01_L003_R1_001.fastq.gz \
../datasets/allen_E01/L8TX_181211_01_E01_S01_L003_R2_001.fastq.gz \

kb count --verbose \
-i ../ref/refdata-gex-mm10-2020-A/kallisto/index.idx \
-g ../ref/refdata-gex-mm10-2020-A/t2g_mm10.txt \
-x 10xv3 \
-o ./allen_F01/ \
-t 50 -m 50G \
-c1 ../ref/refdata-gex-mm10-2020-A/kallisto/cdna_t2c.txt \
-c2 ../ref/refdata-gex-mm10-2020-A/kallisto/intron_t2c.txt \
--workflow lamanno --filter bustools --overwrite --loom \
../datasets/allen_F01/L8TX_181211_01_F01_S01_L003_R1_001.fastq.gz \
../datasets/allen_F01/L8TX_181211_01_F01_S01_L003_R2_001.fastq.gz \


kb count --verbose \
-i ../ref/refdata-gex-mm10-2020-A/kallisto/index.idx \
-g ../ref/refdata-gex-mm10-2020-A/t2g_mm10.txt \
-x 10xv3 \
-o ./allen_G12/ \
-t 50 -m 50G \
-c1 ../ref/refdata-gex-mm10-2020-A/kallisto/cdna_t2c.txt \
-c2 ../ref/refdata-gex-mm10-2020-A/kallisto/intron_t2c.txt \
--workflow lamanno --filter bustools --overwrite --loom \
../datasets/allen_G12/L8TX_181211_01_G12_S01_L003_R1_001.fastq.gz \
../datasets/allen_G12/L8TX_181211_01_G12_S01_L003_R2_001.fastq.gz \

kb count --verbose \
-i ../ref/refdata-gex-mm10-2020-A/kallisto/index.idx \
-g ../ref/refdata-gex-mm10-2020-A/t2g_mm10.txt \
-x 10xv3 \
-o ./allen_H12/ \
-t 50 -m 50G \
-c1 ../ref/refdata-gex-mm10-2020-A/kallisto/cdna_t2c.txt \
-c2 ../ref/refdata-gex-mm10-2020-A/kallisto/intron_t2c.txt \
--workflow lamanno --filter bustools --overwrite --loom \
../datasets/allen_H12/L8TX_181211_01_H12_S01_L003_R1_001.fastq.gz \
../datasets/allen_H12/L8TX_181211_01_H12_S01_L003_R2_001.fastq.gz \

kb count --verbose \
-i ../ref/refdata-gex-mm10-2020-A/kallisto/index.idx \
-g ../ref/refdata-gex-mm10-2020-A/t2g_mm10.txt \
-x 10xv3 \
-o ./allen_F08/ \
-t 50 -m 50G \
-c1 ../ref/refdata-gex-mm10-2020-A/kallisto/cdna_t2c.txt \
-c2 ../ref/refdata-gex-mm10-2020-A/kallisto/intron_t2c.txt \
--workflow lamanno --filter bustools --overwrite --loom \
../datasets/allen_F08/L8TX_190430_01_F08_S01_L003_R1_001.fastq.gz \
../datasets/allen_F08/L8TX_190430_01_F08_S01_L003_R2_001.fastq.gz \

kb count --verbose \
-i ../ref/refdata-gex-mm10-2020-A/kallisto/index.idx \
-g ../ref/refdata-gex-mm10-2020-A/t2g_mm10.txt \
-x 10xv3 \
-o ./allen_G08/ \
-t 50 -m 50G \
-c1 ../ref/refdata-gex-mm10-2020-A/kallisto/cdna_t2c.txt \
-c2 ../ref/refdata-gex-mm10-2020-A/kallisto/intron_t2c.txt \
--workflow lamanno --filter bustools --overwrite --loom \
../datasets/allen_G08/L8TX_190430_01_G08_S01_L003_R1_001.fastq.gz \
../datasets/allen_G08/L8TX_190430_01_G08_S01_L003_R2_001.fastq.gz \
