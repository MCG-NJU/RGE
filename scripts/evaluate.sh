#!/usr/bin/env bash

set -x
set -e
conda activate RGE-env

MODEL_NAME="/path/to/downloaded/RGE"
PROCESSOR_NAME=$MODEL_NAME

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="./outputs/RGE-eval"
fi
if [ -z "$BATCH_SIZE" ]; then
  BATCH_SIZE=32
fi
if [ -z "$MODEL_BACKBONE" ]; then
  MODEL_BACKBONE=qwen2_5_vl
fi

if [ -z "$PROCESSOR_NAME" ]; then
  PROCESSOR_NAME="/path/to/downloaded/RGE"
fi

if [ -z "$IMAGE_RESOLUTION" ]; then
  IMAGE_RESOLUTION=original
fi

if [ -z "$MAX_LEN" ]; then
  MAX_LEN=1280
fi

mkdir -p ${OUTPUT_DIR}

python ./eval_ar.py \
  --processor_name "${PROCESSOR_NAME}" \
  --model_name "${MODEL_NAME}" \
  --encode_output_path "${OUTPUT_DIR}" \
  --max_len ${MAX_LEN} \
  --pooling last --normalize True \
  --dataloader_num_workers 4 \
  --dataset_name "/path/to/MMEB-eval" \
  --image_dir "/path/to/MMEB-eval/image-tasks" \
  --subset_name Wiki-SS-NQ Visual7W-Pointing RefCOCO RefCOCO-Matching ImageNet-1K N24News HatefulMemes SUN397 VOC2007 InfographicsVQA ChartQA A-OKVQA DocVQA OK-VQA Visual7W VisDial CIRR NIGHTS WebQA VisualNews_i2t VisualNews_t2i MSCOCO_i2t MSCOCO_t2i MSCOCO Place365 ImageNet-A ImageNet-R ObjectNet Country211 ScienceQA VizWiz GQA TextVQA OVEN FashionIQ EDIS \
  --instruction_replacements ./src/instruction_replacements-eval.yaml \
  --candidate_replacements ./src/candidate_replacements.yaml \
  --dataset_split test --per_device_eval_batch_size ${BATCH_SIZE} \
  --min_patch_size 256 --max_patch_size 1024 \
  --image_resolution "${IMAGE_RESOLUTION}" \
  --model_backbone "${MODEL_BACKBONE}" \