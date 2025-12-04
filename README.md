# Mano
## Train Stage1
```
CUDA_LAUNCH_BLOCKING=1 python skillmimic/run.py --task SkillMimicHandRand \
--num_envs 4096 \
--episode_length 60 \
--cfg_env skillmimic/data/cfg/mano/mano_stage1_precise_track.yaml \
--cfg_train skillmimic/data/cfg/train/rlg/skillmimic_denseobj.yaml \
--motion_file skillmimic/data/motions/dexgrasp_train_mano/bottle/grasp_higher_kp \
--state_noise_prob 0.2 \
--enable_obj_keypoints \
--enable_ig_scale \
--use_delta_action \
--enable_dof_obs \
--enable_early_termination \
--hand_model mano \
--objnames Bottle \
--headless
```
or you can
```
bash teacher_train_stage1.sh
```

## Train Stage2
```
CUDA_LAUNCH_BLOCKING=1 python skillmimic/run.py --task SkillMimicHandRand \
--num_envs 4096 \
--episode_length 60 \
--cfg_env skillmimic/data/cfg/mano/mano_stage2_noisey_generalize.yaml \
--cfg_train skillmimic/data/cfg/train/rlg/skillmimic_denseobj.yaml \
--motion_file skillmimic/data/motions/dexgrasp_train_mano/bottle/grasp_higher_kp \
--state_noise_prob 0.5 \
--obj_rand_scale \
--enable_obj_keypoints \
--enable_ig_scale \
--use_delta_action \
--enable_dof_obs \
--enable_early_termination \
--hand_model mano \
--objnames Bottle \
--headless
```
or you can
```
bash teacher_train_stage2.sh
```

# Shadow
## Train
```
CUDA_LAUNCH_BLOCKING=1 python skillmimic/run.py --task SkillMimicHandRand \
--num_envs 4096 \
--episode_length 60 \
--cfg_env skillmimic/data/cfg/shadow/shadow_stage1_precise_track.yaml \
--cfg_train skillmimic/data/cfg/train/rlg/skillmimic_denseobj.yaml \
--motion_file skillmimic/data/motions/dexgrasp_train_shadow/bottle/grasp_higher_kp \
--state_noise_prob 0.2 \
--enable_obj_keypoints \
--enable_ig_scale \
--enable_dof_obs \
--use_delta_action \
--enable_early_termination \
--hand_model shadow \
--objnames Bottle \
--headless
```
## Test
Note that: 之前为了加速训练使用了res_action(学习相对于reference的action残差)，新训的model还是用delta_action吧
```
CUDA_LAUNCH_BLOCKING=1 python skillmimic/run.py --task SkillMimicHandRand \
--test \
--num_envs 1 \
--episode_length 180 \
--cfg_env skillmimic/data/cfg/shadow/shadow_stage1_precise_track.yaml \
--cfg_train skillmimic/data/cfg/train/rlg/skillmimic_denseobj.yaml \
--motion_file skillmimic/data/motions/dexgrasp_train_shadow/bottle/grasp_higher_kp \
--state_init 2 \
--enable_obj_keypoints \
--enable_ig_scale \
--enable_dof_obs \
--use_res_action \
--hand_model shadow \
--objnames Bottle \
--checkpoint checkpoint/shadow/shadow_bottle_grasp-move-place.pth
```

# Allegro
## Train
```
CUDA_LAUNCH_BLOCKING=1 python skillmimic/run.py --task SkillMimicHandRand \
--num_envs 4096 \
--episode_length 60 \
--cfg_env skillmimic/data/cfg/allegro/allegro_stage1_precise_track.yaml \
--cfg_train skillmimic/data/cfg/train/rlg/skillmimic_denseobj.yaml \
--motion_file skillmimic/data/motions/dexgrasp_train_allegro/bottle/grasp_higher_kp \
--state_noise_prob 0.2 \
--enable_obj_keypoints \
--enable_ig_scale \
--enable_dof_obs \
--use_delta_action \
--enable_early_termination \
--hand_model allegro \
--objnames Bottle \
--headless \
```
## Test
Note that: 这个模型是hokwai训的，她的observation没有和我align上，所以Inference出的性能较差
```
CUDA_LAUNCH_BLOCKING=1 python skillmimic/run.py --task SkillMimicHandRand \
--test \
--num_envs 1 \
--episode_length 180 \
--cfg_env skillmimic/data/cfg/allegro/allegro_stage1_precise_track.yaml \
--cfg_train skillmimic/data/cfg/train/rlg/skillmimic_denseobj.yaml \
--motion_file skillmimic/data/motions/dexgrasp_train_allegro/bottle/grasp_higher_kp \
--state_init 2 \
--enable_obj_keypoints \
--enable_ig_scale \
--enable_dof_obs \
--use_delta_action \
--hand_model allegro \
--objnames Bottle \
--checkpoint checkpoint/allegro/allegro_bottle_grasp-move-place.pth
```
