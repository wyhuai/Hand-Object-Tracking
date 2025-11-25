#!/bin/bash

# 要使用的GPU设备列表
GPUS=(0 1 2 3 4 5)

# 对象列表
OBJECTS=("airplane" "ball" "book" "eagle" "flyingdisc" "hand")

# 检查对象数量和GPU数量是否匹配
if [ ${#OBJECTS[@]} -ne ${#GPUS[@]} ]; then
    echo "错误: 对象数量 (${#OBJECTS[@]}) 和 GPU数量 (${#GPUS[@]}) 不匹配!"
    exit 1
fi

# 创建日志目录
mkdir -p logs

echo "开始在多卡上运行任务..."
echo "GPU列表: ${GPUS[@]}"
echo "对象列表: ${OBJECTS[@]}"
echo ""

# 在每个GPU上运行命令，每个GPU处理一个对象
for i in "${!GPUS[@]}"; do
    gpu=${GPUS[$i]}
    obj=${OBJECTS[$i]}
    obj_lower=$(echo "$obj" | tr '[:upper:]' '[:lower:]')
    # 确保物体名称首字母大写
    obj_capitalized="$(tr '[:lower:]' '[:upper:]' <<< ${obj:0:1})${obj:1}"
    
    echo "=== 在 GPU $gpu 上运行对象: $obj ($obj_lower) ==="
    
    # 查找包含物体名称的检查点文件
    checkpoint_file=$(find ./checkpoint/teachers -name "*${obj_lower}*.pth" | head -1)
    
    if [ -z "$checkpoint_file" ]; then
        echo "错误: 找不到对象 $obj 的检查点文件!"
        echo "跳过该对象 $obj"
        echo ""
        continue
    fi
    
    echo "使用检查点文件: $checkpoint_file"
    
    # 构建完整的命令
    CMD="CUDA_VISIBLE_DEVICES=$gpu CUDA_LAUNCH_BLOCKING=1 python -u skillmimic/run.py \
        --task SkillMimic2BallPlayRandInd \
        --num_envs 4096 \
        --episode_length 60 \
        --cfg_env skillmimic/data/cfg/skillmimic_wrist.yaml \
        --cfg_train skillmimic/data/cfg/train/rlg/skillmimic_denseobj.yaml \
        --motion_file \"skillmimic/data/motions/dexgrasp_train_mano_20obj/$obj_lower\" \
        --state_noise_prob 0.5 \
        --enable_wrist_local_obs \
        --enable_obj_keypoints \
        --enable_ig_scale \
        --use_delta_action \
        --enable_dof_obs \
        --obj_rand_scale \
        --enable_early_termination \
        --headless \
        --objnames \"$obj_capitalized\" \
        --resume_from \"$checkpoint_file\" \
        --max_iterations 10000 \
        --experiment \"skillmimic_${obj_lower}_gpu${gpu}_finetune\""
    
    # 输出将要运行的命令
    echo "执行的命令:"
    echo "$CMD"
    echo "日志文件: logs/skillmimic_${obj_lower}_gpu${gpu}_finetune.log"
    echo ""
    
    # 实际运行命令
    eval $CMD > "logs/skillmimic_${obj_lower}_gpu${gpu}_finetune.log" 2>&1 &

    PID=$!
    echo "启动进程 $PID 在 GPU $gpu 上处理对象 $obj"
    echo "---"
    sleep 5  # 延迟5秒，避免资源冲突
done

echo "所有进程已启动!"
echo "使用 'jobs' 查看运行状态"
echo "使用 'tail -f logs/skillmimic_*.log' 查看日志"
echo "使用 'pkill -f skillmimic' 停止所有进程"

# 等待所有后台进程完成
wait
echo "所有任务完成!"