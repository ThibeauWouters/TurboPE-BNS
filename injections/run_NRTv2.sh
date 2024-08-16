#!/bin/bash

for i in {1..10}
do
    echo "==================== Run #$i ===================="
    
    python injection_recovery.py \
    --outdir ./outdir_NRTv2_new_taper/ \
    --relative-binning-binsize 200 \
    --stopping-criterion-global-acc 0.10 \
    --waveform-approximant IMRPhenomD_NRTidalv2
done
