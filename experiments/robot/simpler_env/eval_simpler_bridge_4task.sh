
ckpt_path="/mnt/project_rlinf/jlchen/code/UniVLA/vla_scripts/world_vla_log/0127_222333+weights+bridge+lr-0.0001+frzVis+frzLLM+unfrzLast4+frzEmb--finetune_bridge_vldit/checkpoints/step-050000"
num_episodes=1

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python real2sim_eval_maniskill3.py \
    --model="latent_world_vla" -e "PutSpoonOnTableClothInScene-v1" -s 0 --num-episodes ${num_episodes} --num-envs 1 \
    --ckpt_path ${ckpt_path} \
    
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python real2sim_eval_maniskill3.py \
    --model="latent_world_vla" -e "PutCarrotOnPlateInScene-v1" -s 0 --num-episodes ${num_episodes} --num-envs 1 \
    --ckpt_path ${ckpt_path} \

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python real2sim_eval_maniskill3.py \
    --model="latent_world_vla" -e "StackGreenCubeOnYellowCubeBakedTexInScene-v1" -s 0 --num-episodes ${num_episodes} --num-envs 1 \
    --ckpt_path ${ckpt_path} \

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python real2sim_eval_maniskill3.py \
    --model="latent_world_vla" -e "PutEggplantInBasketScene-v1" -s 0 --num-episodes ${num_episodes} --num-envs 1 \
    --ckpt_path ${ckpt_path} \

