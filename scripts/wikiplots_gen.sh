SAVE_DIR=models
DATA_NAME=wikiplots
DATA_PATH=data/${DATA_NAME}.json
MODEL_NAME=discoDVT_${DATA_NAME}
PRETRAINED_MODEL_DIR=${SAVE_DIR}/bart-base


python src/main.py \
--dataset_path ${DATA_PATH} \
--model_checkpoint ${SAVE_DIR}/${MODEL_NAME} \
--save_dir ${SAVE_DIR} \
--model_name ${MODEL_NAME} \
--generate_with_code \
--max_src_len 16 \
--max_tgt_len 512 \
--num_cnn_layers 3 \
--valid_batch_size 4 \
--device cuda:1 \
--do_generate \
--rep_penalty 1.0 \
--min_length 100 \
--latent_vocab_size 256 \
--test_sample_N 1000 \
--topp 0.9 \
--temperature 1.0 \
| tee ${SAVE_DIR}/${MODEL_NAME}/inference.log
wait $!
