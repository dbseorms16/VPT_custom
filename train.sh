# %%bash
# launch final training with five random seeds for VTAB-dmlab, sun397 and eurosat. The hyperparameters are the same from our paper.
model_root=./
data_path=./dataset/data
output_dir=./result
        
# vtab-structured: dmlab
# base_lr = 1.0
# lr = base_lr / 256 * cfg.DATA.BATCH_SIZE
# for seed in "42" "44" "82" "100" "800"; do
#     python train.py \
#         --config-file configs/prompt/cub.yaml \
#         MODEL.TYPE "vit" \
#         DATA.BATCH_SIZE "64" \
#         MODEL.PROMPT.NUM_TOKENS "100" \
#         MODEL.PROMPT.DEEP "True" \
#         MODEL.PROMPT.DROPOUT "0.1" \
#         DATA.FEATURE "sup_vitb16_imagenet21k" \
#         DATA.NAME "vtab-dmlab" \
#         DATA.NUMBER_CLASSES "6" \
#         SOLVER.BASE_LR "0.25" \
#         SOLVER.WEIGHT_DECAY "0.001" \
#         SEED ${seed} \
#         MODEL.MODEL_ROOT "${model_root}" \
#         DATA.DATAPATH "${data_path}" \
#         OUTPUT_DIR "${output_dir}/seed${seed}" 
# done

# # vtab-natural: sun397
# # base_lr = 25
# # lr = base_lr / 256 * cfg.DATA.BATCH_SIZE
# for seed in "42" "44" "82" "100" "800"; do
#     python train.py \
#         --config-file configs/prompt/cub.yaml \
#         MODEL.TYPE "vit" \
#         DATA.BATCH_SIZE "128" \
#         MODEL.PROMPT.NUM_TOKENS "5" \
#         MODEL.PROMPT.DEEP "True" \
#         MODEL.PROMPT.DROPOUT "0.1" \
#         DATA.FEATURE "sup_vitb16_imagenet21k" \
#         DATA.NAME "vtab-sun397" \
#         DATA.NUMBER_CLASSES "397" \
#         SOLVER.BASE_LR "12.5" \
#         SOLVER.WEIGHT_DECAY "0.0001" \
#         SOLVER.TOTAL_EPOCH "100" \
#         SEED ${seed} \
#         MODEL.MODEL_ROOT "${model_root}" \
#         DATA.DATAPATH "${data_path}" \
#         OUTPUT_DIR "${output_dir}/seed${seed}" 
# done

# vtab-specialized: vtab-eurosat
# base_lr = 1
# lr = base_lr / 256 * cfg.DATA.BATCH_SIZE

# for seed in "42" "44" "82" "100" "800"; do
#     python train.py \
#         --config-file configs/prompt/cub.yaml \
#         MODEL.TYPE "vit" \
#         DATA.BATCH_SIZE "64" \
#         MODEL.PROMPT.NUM_TOKENS "100" \
#         MODEL.PROMPT.DEEP "True" \
#         MODEL.PROMPT.DROPOUT "0.1" \
#         DATA.FEATURE "sup_vitb16_imagenet21k" \
#         DATA.NAME "vtab-eurosat" \
#         DATA.NUMBER_CLASSES "10" \
#         SOLVER.BASE_LR "0.25" \
#         SOLVER.WEIGHT_DECAY "0.001" \
#         SOLVER.TOTAL_EPOCH "100" \
#         SEED ${seed} \
#         MODEL.MODEL_ROOT "${model_root}" \
#         DATA.DATAPATH "${data_path}" \
#         OUTPUT_DIR "${output_dir}/seed${seed}" 
# done

for seed in "42" "44" "82" "100" "800"; do
    python -m torch.distributed.launch --nproc_per_node=7 
        train.py \
        --config-file configs/prompt/custom.yaml \
        MODEL.TYPE "sam" \
        DATA.BATCH_SIZE "1" \
        MODEL.PROMPT.NUM_TOKENS "100" \
        MODEL.PROMPT.DEEP "True" \
        MODEL.PROMPT.DROPOUT "0.1" \
        DATA.FEATURE "sup_vitb16_imagenet21k" \
        DATA.NAME "vtab-eurosat" \
        DATA.NUMBER_CLASSES "10" \
        SOLVER.BASE_LR "0.25" \
        SOLVER.WEIGHT_DECAY "0.001" \
        SOLVER.TOTAL_EPOCH "100" \
        SEED ${seed} \
        MODEL.MODEL_ROOT "${model_root}" \
        DATA.DATAPATH "${data_path}" \
        OUTPUT_DIR "${output_dir}/seed${seed}" 
done