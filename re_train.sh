#!/usr/bin/env bash
python3 training/generate_dataset.py      
python3 training/train_model.py        
python3 training/convert_ptq.py           
python3 training/prune_quantise.py    
cp training/models/model_pruned_int8.tflite inference/model.tflite
./training/verify_model_sync.sh  
python3 monitoring/build_reference_dist.py
python3 monitoring/drift_demo_offline.py