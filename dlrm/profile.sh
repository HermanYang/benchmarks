#!/bin/bash

#python dlrm_s_pytorch.py \
#  --arch-sparse-feature-size 64 \
#  --arch-embedding-size 9980333-36084-17217-7378-20134-3-7112-1442-61-9758201-1333352-313829-10-2208-11156-122-4-970-14-9994222-7267859-9946608-415421-12420-101-36 \
#  --arch-mlp-bot 13-512-256-64 \
#  --arch-mlp-top 415-512-512-256-1 \
#  --arch-interaction-op dot \
#  --num-batches 10 \
#  --max-ind-range 10000000 \
#  --memory-map \
#  --mini-batch-size 2048 \
#  --inference-only \
#  --use-gpu \
#  --tensor-board-filename tensorboard_logs 
  # --print-time \
  # --print-wall-time \
  # --debug-mode \
  # --enable-profiling \


# ncu -f -o dlrm --set full --target-processes all \
# python dlrm_s_pytorch.py \
#   --arch-sparse-feature-size 64 \
#   --arch-embedding-size 9980333-36084-17217-7378-20134-3-7112-1442-61-9758201-1333352-313829-10-2208-11156-122-4-970-14-9994222-7267859-9946608-415421-12420-101-36 \
#   --arch-mlp-bot 13-512-256-64 \
#   --arch-mlp-top 415-512-512-256-1 \
#   --arch-interaction-op dot \
#   --num-batches 10 \
#   --max-ind-range 10000000 \
#   --memory-map \
#   --mini-batch-size 2048 \
#   --inference-only \
#   --use-gpu \
#   --tensor-board-filename tensorboard_logs 

# nsys profile --kill=none -o dlrm -f true \
# python dlrm_s_pytorch.py \
#   --arch-sparse-feature-size 64 \
#   --arch-embedding-size 9980333-36084-17217-7378-20134-3-7112-1442-61-9758201-1333352-313829-10-2208-11156-122-4-970-14-9994222-7267859-9946608-415421-12420-101-36 \
#   --arch-mlp-bot 13-512-256-64 \
#   --arch-mlp-top 415-512-512-256-1 \
#   --arch-interaction-op dot \
#   --num-batches 10 \
#   --max-ind-range 10000000 \
#   --memory-map \
#   --mini-batch-size 2048 \
#   --inference-only \
#   --use-gpu \
#   --tensor-board-filename tensorboard_logs 

python dlrm_s_pytorch.py \
  --arch-sparse-feature-size 64 \
  --arch-embedding-size 9980333-36084-17217-7378-20134-3-7112-1442-61-9758201-1333352-313829-10-2208-11156-122-4-970-14-9994222-7267859-9946608-415421-12420-101-36 \
  --arch-mlp-bot 13-512-256-64 \
  --arch-mlp-top 415-512-512-256-1 \
  --arch-interaction-op dot \
  --num-batches 10 \
  --max-ind-range 10000000 \
  --memory-map \
  --mini-batch-size 1 \
  --inference-only  \
  --use-gpu