#!/bin/bash

# 要使用的GPU设备列表
GPUS=(1)

# 对象列表 - 根据您的实际对象修改
OBJECTS=("Bottle")

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
    LATEST_OUTPUT_DIR=$(ls -td output/skillmimic_${obj_lower}_gpu*_new_* 2>/dev/null | head -1)
    
    RESUME_ARG=""
    
    if [ -n "$LATEST_OUTPUT_DIR" ]; then
        # 在找到的输出目录的nn子目录中查找最新的pth文件
        LATEST_CKPT=$(ls -t ${LATEST_OUTPUT_DIR}/nn/*.pth 2>/dev/null | head -1)
        if [ -n "$LATEST_CKPT" ]; then
            RESUME_ARG="--resume_from \"$LATEST_CKPT\""
            echo "找到checkpoint: $LATEST_CKPT"
        fi
    fi
    echo "=== 在 GPU $gpu 上运行对象: $obj ($obj_lower) ==="
    
    # 构建完整的命令 - 使用 -u 参数禁用输出缓冲
    CMD="CUDA_VISIBLE_DEVICES=$gpu CUDA_LAUNCH_BLOCKING=1 python -u hot/run.py \
        --task SkillMimicHandRand \
        --num_envs 4096 \
        --episode_length 60 \
        --cfg_env hot/data/cfg/mano/mano_stage2_noisey_generalize.yaml \
        --cfg_train hot/data/cfg/train/rlg/skillmimic_denseobj.yaml \
        --motion_file \"hot/data/motions/dexgrasp_train_mano_gmp/$obj_lower\" \
        --state_noise_prob 0.5 \
        --obj_rand_scale \
        --enable_obj_keypoints \
        --enable_ig_scale \
        --use_delta_action \
        --enable_dof_obs \
        --enable_early_termination \
        --headless \
        --objnames \"$obj\" \
        --experiment \"skillmimic_${obj_lower}_gpu${gpu}_finetune\"\
        $RESUME_ARG  \
        --max_iterations 10000 "
    
    # 输出将要运行的命令
    echo "执行的命令:"
    echo "$CMD"
    echo "日志文件: logs/skillmimic_${obj_lower}_gpu${gpu}.log"
    echo ""
    
    # 实际运行命令
    eval $CMD > "logs/skillmimic_${obj_lower}_gpu${gpu}.log" 2>&1 &
    
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
