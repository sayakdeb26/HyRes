#!/bin/bash
source ~/venvs/rosgpu_isolated/bin/activate
export MKL_THREADING_LAYER=GNU
python3 /home/sayak/HyRes/run_phase2_vlm_experiment.py
