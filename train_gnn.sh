python3 train_gnn_downscaler.py \
  --lat-min 25.0 --lat-max 45.0 \
  --lon-min -100.0 --lon-max -80.0 \
  --nodes 350 \
  --rollout-steps 3 \
  --epochs 30 \
  --save-weights ./checkpoints/gulf_climate_downscaler.pt
