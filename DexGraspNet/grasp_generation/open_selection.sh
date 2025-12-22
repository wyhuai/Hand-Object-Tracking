#!/bin/bash

# Read indices from save_id.txt and open the corresponding URLs
while IFS= read -r index; do
  open "http://localhost:5000/results/$index"
done < /DexGraspNet/data/experiments_stick_02/exp_32/saved_ids.txt