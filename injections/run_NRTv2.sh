# Run with the taper problem to see if we can fix it at some point
python injection_recovery.py \
    --outdir ./debug_taper/ \
    --relative-binning-binsize 200 \
    --stopping-criterion-global-acc 0.10 \
    --waveform-approximant IMRPhenomD_NRTidalv2
