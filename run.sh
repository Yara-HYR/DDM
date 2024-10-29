#!/bin/bash



export PYTHONPATH=$PYTHONPATH:/data/XXXX-4/RL/RL_envs/carla_0.9.13/PythonAPI/carla/dist/carla-0.9.13-py3.5-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:/data/XXXX-4/RL/RL_envs/carla_0.9.13/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:/data/XXXX-4/RL/RL_envs/carla_0.9.13/PythonAPI

SUIT=carla

DOMAIN=highway_normal
WEATHER=normal

# DOMAIN=highway
# WEATHER=midnight

# DOMAIN=highway_normal
# WEATHER=hard_rain

# DOMAIN=highway_blinding
# WEATHER=blinding


SPEED=60
AGENT=cross_pre_incon

PERCEPTION=RGB-Frame+DVS-Voxel-Grid
# +LiDAR-BEV + Depth-Frame


ENCODER=pixelCat

DECODER=identity

RPC_PORT=8632
TM_PORT=18632
RPC_PORT_EVAL=7632
TM_PORT_EVAL=17632

CUDA_DEVICE=1

SEED=111

UNIQUE_ID=${SUIT}+${DOMAIN}+${WEATHER}+${SPEED}+${AGENT}+${PERCEPTION}+${ENCODER}+${DECODER}+${SEED}
LOGFILE=./logs/${UNIQUE_ID}.log
WORKDIR=./logs/${UNIQUE_ID}



echo ${UNIQUE_ID}
mkdir -p ${WORKDIR}


CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python -u train_testm.py \
    --work_dir ${WORKDIR}$ \
    --suit carla \
    --domain_name ${DOMAIN} \
    --selected_weather ${WEATHER} \
    --agent ${AGENT} \
    --perception_type ${PERCEPTION} \
    --encoder_type ${ENCODER} \
    --decoder_type ${DECODER} \
    --action_model_update_freq 1 \
    --transition_reward_model_update_freq 1 \
    --carla_rpc_port ${RPC_PORT} \
    --carla_tm_port ${TM_PORT} \
    --carla_rpc_port_eval ${RPC_PORT_EVAL} \
    --carla_tm_port_eval ${TM_PORT_EVAL} \
    --carla_timeout 30 \
    --frame_skip 1 \
    --init_steps 1000 \
    --max_episode_steps 1000 \
    --rl_image_size 128 \
    --num_cameras 1 \
    --actor_lr 1e-3 \
    --critic_lr 1e-3 \
    --encoder_lr 1e-3 \
    --decoder_lr 1e-3 \
    --replay_buffer_capacity 10000 \
    --batch_size 128 \
    --EVAL_FREQ_EPISODE 200 \
    --EVAL_FREQ_STEP 50000 \
    --num_eval_episodes 10 \
    --save_tb \
    --do_metrics \
    --seed ${SEED}    >${LOGFILE} 2>&1 &
    # 
    # 
