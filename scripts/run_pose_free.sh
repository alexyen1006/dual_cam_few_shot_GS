#!/bin/bash

# Change the absolute path first!
GPU_ID=1
DATA_ROOT_DIR="/home/yenalex/dust3R_finetuning/dataset/ZoomGS_dataset"
OUTPUT_DIR="output"
DATASET="ZoomGS"
N_VIEW=6
MODEL_PATH=./${OUTPUT_DIR}/${DATASET}/${SCENE}/${N_VIEW}_views
gs_train_iter=6000

# Main loop
# total_tasks= ${#N_VIEWS[@]} * ${#gs_train_iter[@]}
current_task=0
echo "SOURCE_PATH: $DATA_ROOT_DIR"
#for SCENE in "$DATA_ROOT_DIR"/*/ ; do
for SCENE in 07 08 09 10; do
    SCENE=$(basename "$SCENE")
    echo "SCENE: $SCENE"

    current_task=$((current_task + 1))
    echo "Processing task $current_task / $total_tasks"

    SOURCE_PATH=${DATA_ROOT_DIR}/${SCENE}
    GT_POSE_PATH=${DATA_ROOT_DIR}/${DATASET}/${SCENE}/
    IMAGE_PATH=${SOURCE_PATH}images
    MODEL_PATH=./${OUTPUT_DIR}/${DATASET}/${SCENE}/${N_VIEW}_views

    # Create necessary directories
    # mkdir -p ${MODEL_PATH}

    # echo "======================================================="
    # echo "Starting process: ${DATASET}/${SCENE} (${N_VIEW} views/${gs_train_iter} iters) on GPU ${GPU_ID}"
    # echo "======================================================="


    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Co-visible Global Geometry Initialization..."
    CUDA_VISIBLE_DEVICES=${GPU_ID} python -W ignore ./init_geo.py \
    -s ${SOURCE_PATH} \
    -m ${MODEL_PATH} \
    --n_views ${N_VIEW} \
    --co_vis_dsp \
    --conf_aware_ranking \
    #####--with_GT_focal\
       ##--focal_avg \
    
    
    # # > ${MODEL_PATH}/01_init_geo.log 2>&1
    # # echo "[$(date '+%Y-%m-%d %H:%M:%S')] Co-visible Global Geometry Initialization completed. Log saved in ${MODEL_PATH}/01_init_geo.log"



    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting pretraining by uw..."
    CUDA_VISIBLE_DEVICES=${GPU_ID} python ./train.py \
    -s ${SOURCE_PATH} \
    -m ${MODEL_PATH} \
    --n_views ${N_VIEW} \
    -r 1 \
    --iterations ${gs_train_iter} \
    --pp_optimizer \
    --optim_pose \
    --port 6014 \
    --stage "uw_pretrain"\
    
    # > ${MODEL_PATH}/02_train.log 2>&1
    # echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training completed. Log saved in ${MODEL_PATH}/02_train.log"
    

    # echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting training uw to wide..."
    # CUDA_VISIBLE_DEVICES=${GPU_ID} python ./train.py \
    # -s ${SOURCE_PATH} \
    # -m ${MODEL_PATH} \
    # --n_views ${N_VIEW} \
    # -r 1 \
    # --iterations ${gs_train_iter} \
    # --optim_pose \
    # --port 6014 \
    # --stage "uw2wide"\
    ##--fsgs_loss \
    ## --with_identity_loss \
    ## --identity_loss_weight 0.3 \
    ## --fsgs_loss \
    ##> ${MODEL_PATH}/02_train.log 2>&1
    #echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training completed. Log saved in ${MODEL_PATH}/02_train.log"
    
    
 
    # echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting rendering training views..."
    # CUDA_VISIBLE_DEVICES=${GPU_ID} python ./render.py \
    # -s ${SOURCE_PATH} \
    # -m ${MODEL_PATH} \
    # -r 1 \
    # --n_views ${N_VIEW} \
    # --iterations ${gs_train_iter} \
    # # # ####> ${MODEL_PATH}/03_render_train.log 2>&1
    # # # echo "[$(date '+%Y-%m-%d %H:%M:%S')] Rendering completed. Log saved in ${MODEL_PATH}/03_render_train.log"


    # echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting rendering testing views..."
    # CUDA_VISIBLE_DEVICES=${GPU_ID} python ./render.py \
    # -s ${SOURCE_PATH} \
    # -m ${MODEL_PATH} \
    # -r 1 \
    # --n_views ${N_VIEW} \
    # --iterations ${gs_train_iter} \
    # --eval \
    # ###--test_fps \
    # #> ${MODEL_PATH}/04_render_test.log 2>&1
    # # echo "[$(date '+%Y-%m-%d %H:%M:%S')] Rendering completed. Log saved in ${MODEL_PATH}/04_render_test.log"
    


    # # echo "[$(date '+%Y-%m-%d %H:%M:%S')] Calculating metrics..."
    # CUDA_VISIBLE_DEVICES=${GPU_ID} python ./metrics.py \
    # -s ${SOURCE_PATH} \
    # -m ${MODEL_PATH} \
    # --n_views ${N_VIEW} \
    # #> ${MODEL_PATH}/05_metrics.log 2>&1
    # echo "[$(date '+%Y-%m-%d %H:%M:%S')] Metrics calculation completed. Log saved in ${MODEL_PATH}/05_metrics.log"

    # echo "======================================================="
    # echo "Task completed: ${DATASET}/${SCENE} (${N_VIEW} views/${gs_train_iter} iters) on GPU ${GPU_ID}"
    # echo "======================================================="
    
done


# Wait for all background tasks to complete
wait

echo "======================================================="
echo "All tasks completed! Processed $total_tasks tasks in total."
echo "======================================================="
