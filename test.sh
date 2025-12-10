#!/bin/bash
set -euo pipefail

# ================= Configuration =================

# [Modified] Define unified Checkpoint path here
CHECKPOINT_PATH="checkpoint/multiobj_student_checkpoint/distill_4layer_15obj_4096.pth"

# Define object list
objects=(
    "Basket"
    "Book"
    "Bowl"
    "Bowlround"
    "Eagle"
    "flower"
    "hand"
    "headphones"
    "pencil"
    "pizza"
    "spidermanfigure"
    "stapler"
    "sterilite"
    "wineglass"
)

# ===========================================

# Create log directory
mkdir -p logs
mkdir -p logs/test_result

# Create result output file
timestamp=$(date +%Y%m%d_%H%M%S)
output_file="logs/test_result/test_results_${timestamp}.txt"
summary_file="logs/test_result/test_summary_${timestamp}.csv"

echo "Test Results - $(date)" > "$output_file"
echo "Using Checkpoint: $CHECKPOINT_PATH" >> "$output_file"
echo "========================================" >> "$output_file"
echo "" >> "$output_file"

# Create summary table (CSV)
echo "objname,grasp_MoveMetric,place_MoveMetric,move_MoveMetric" > "$summary_file"

# Iterate through each object
for obj in "${objects[@]}"; do
    # Convert to lowercase initial (for motion_file path)
    obj_lower=$(echo "$obj" | awk '{print tolower(substr($0,1,1)) substr($0,2)}')

    # Convert to uppercase initial (for objnames argument)
    obj_upper=$(echo "$obj" | awk '{print toupper(substr($0,1,1)) substr($0,2)}')

    echo "========================================" | tee -a "$output_file"
    echo "Processing object: $obj (Lower: $obj_lower, Upper: $obj_upper)" | tee -a "$output_file"
    echo "Time: $(date)" | tee -a "$output_file"
    echo "========================================" | tee -a "$output_file"
    echo "" | tee -a "$output_file"

    # Independent log files for each object
    grasp_log="logs/${timestamp}_${obj_lower}_grasp.log"
    place_log="logs/${timestamp}_${obj_lower}_place.log"
    move_log="logs/${timestamp}_${obj_lower}_move.log"

    ###############################
    # Command 1: grasp_kp
    ###############################
    echo "[$obj] Executing Command 1: grasp_kp" | tee -a "$output_file"
    echo "----------------------------------------" | tee -a "$output_file"

    # [Modified] Use $CHECKPOINT_PATH variable
    grasp_command="DRI_PRIME=1 CUDA_VISIBLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=1 \
    python hot/run.py \
        --test \
        --task SkillMimicHandRand \
        --num_envs 1024 \
        --cfg_env hot/data/cfg/mano/mano_stage1_precise_track.yaml \
        --cfg_train hot/data/cfg/train/rlg/skillmimic_large_mlp.yaml\
        --enable_obj_keypoints \
        --enable_ig_scale \
        --checkpoint \"$CHECKPOINT_PATH\" \
        --state_init 2 \
        --motion_file \"hot/data/motions/dexgrasp_train_mano_20obj/${obj_lower}/grasp_kp\" \
        --objnames \"$obj_upper\" \
        --headless \
        --use_delta_action \
        --enable_dof_obs"

    echo "Executing command: $grasp_command" | tee -a "$output_file" "$grasp_log"
    echo "" | tee -a "$output_file" "$grasp_log"

    # Actually execute command
    eval $grasp_command 2>&1 | tee -a "$grasp_log" | tee -a "$output_file"

    echo "" | tee -a "$output_file"

    ###############################
    # Command 2: place_higher_kp
    ###############################
    echo "[$obj] Executing Command 2: place_higher_kp" | tee -a "$output_file"
    echo "----------------------------------------" | tee -a "$output_file"

    # [Modified] Use $CHECKPOINT_PATH variable
    place_command="DRI_PRIME=1 CUDA_VISIBLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=1 \
    python hot/run.py \
        --test \
        --task SkillMimicHandRand \
        --num_envs 1024 \
        --cfg_env hot/data/cfg/mano/mano_stage1_precise_track.yaml \
        --cfg_train hot/data/cfg/train/rlg/skillmimic_large_mlp.yaml\
        --enable_obj_keypoints \
        --enable_ig_scale \
        --checkpoint \"$CHECKPOINT_PATH\" \
        --state_init 2 \
        --motion_file \"hot/data/motions/dexgrasp_train_mano_20obj/${obj_lower}/place_higher_kp\" \
        --objnames \"$obj_upper\" \
        --headless \
        --use_delta_action \
        --enable_dof_obs \
        --episode_length 220"

    echo "Executing command: $place_command" | tee -a "$output_file" "$place_log"
    echo "" | tee -a "$output_file" "$place_log"

    eval $place_command 2>&1 | tee -a "$place_log" | tee -a "$output_file"

    echo "" | tee -a "$output_file"

    ###############################
    # Command 3: move_higher_kp
    ###############################
    echo "[$obj] Executing Command 3: move_higher_kp" | tee -a "$output_file"
    echo "----------------------------------------" | tee -a "$output_file"

    # [Modified] Use $CHECKPOINT_PATH variable
    move_command="DRI_PRIME=1 CUDA_VISIBLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=1 \
    python hot/run.py \
        --test \
        --task SkillMimicHandRand \
        --num_envs 1024 \
        --cfg_env hot/data/cfg/mano/mano_stage1_precise_track.yaml \
        --cfg_train hot/data/cfg/train/rlg/skillmimic_large_mlp.yaml\
        --enable_obj_keypoints \
        --enable_ig_scale \
        --checkpoint \"$CHECKPOINT_PATH\" \
        --state_init 2 \
        --motion_file \"hot/data/motions/dexgrasp_train_mano_20obj/${obj_lower}/move_higher_kp\" \
        --objnames \"$obj_upper\" \
        --headless \
        --use_delta_action \
        --enable_dof_obs"

    echo "Executing command: $move_command" | tee -a "$output_file" "$move_log"
    echo "" | tee -a "$output_file" "$move_log"

    eval $move_command 2>&1 | tee -a "$move_log" | tee -a "$output_file"

    echo "" | tee -a "$output_file"

    ###################################
    # Extract metrics from logs and write to total results
    ###################################

    echo "[$obj] Metric Summary:" | tee -a "$output_file"
    grep -E "(av reward|av steps|ObjPosError|ObjRotError|KeyPosError|MoveMetric)" "$grasp_log" | tee -a "$output_file" || true
    grep -E "(av reward|av steps|ObjPosError|ObjRotError|KeyPosError|MoveMetric)" "$place_log" | tee -a "$output_file" || true
    grep -E "(av reward|av steps|ObjPosError|ObjRotError|KeyPosError|MoveMetric)" "$move_log" | tee -a "$output_file" || true
    echo "" | tee -a "$output_file"

    # Extract MoveMetric
    grasp_metric=$(grep "MoveMetric" "$grasp_log" | tail -n1 | awk '{print $NF}' || echo "NA")
    place_metric=$(grep "MoveMetric" "$place_log" | tail -n1 | awk '{print $NF}' || echo "NA")
    move_metric=$(grep "MoveMetric" "$move_log"  | tail -n1 | awk '{print $NF}' || echo "NA")

    # Write to CSV summary table
    echo "${obj_upper},${grasp_metric},${place_metric},${move_metric}" >> "$summary_file"

    echo "Finished object: $obj"
    echo ""
done

echo "========================================" >> "$output_file"
echo "All tests completed - $(date)" >> "$output_file"
echo "Results saved to: $output_file" >> "$output_file"
echo "Metrics summary saved to: $summary_file" >> "$output_file"

echo "All tests completed."
echo "Detailed log:   $output_file"
echo "Summary table:  $summary_file"