#!/bin/bash
# .  ~/venv/smp_env/bin/activate
# python3 -m pip install -r requirements.txt

# Download dataset

DATA_DIR="${PWD}/data"
NUM_EPOCHS=50

BATCH=64

if [[ ! -d "${DATA_DIR}/ADEChallengeData2016" ]]; then
   mkdir -p ${DATA_DIR}
   wget -O ${DATA_DIR}/ADEChallengeData2016.zip http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
   unzip ${DATA_DIR}/ADEChallengeData2016.zip -d ${DATA_DIR}
   rm ${DATA_DIR}/ADEChallengeData2016.zip
   echo "Dataset downloaded."
else
   echo "Dataset already downloaded"
fi


# Train the model
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 python3 main.py \
  --data=${DATA_DIR} \
  --epochs=${NUM_EPOCHS} \
  --batch=${BATCH} \
  --lr=1e-4
