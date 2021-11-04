SAVE_DIR=models
DATA_NAME=wikiplots
DATA_PATH=data/${DATA_NAME}.json
MODEL_NAME=discoDVT_${DATA_NAME}

python src/main.py \
--dataset_path ${DATA_PATH} \
--model_checkpoint ${SAVE_DIR}/${MODEL_NAME} \
--save_dir ${SAVE_DIR} \
--model_name ${MODEL_NAME} \
--relation_num 20 \
--max_src_len 16 \
--max_tgt_len 512 \
--valid_batch_size 16 \
--train_batch_size 16 \
--latent_vocab_size 256 \
--num_cnn_layers 3 \
--device cuda:1 \
--decode_code ${1} \
--test_sample_N 1000 \
| tee ${SAVE_DIR}/${MODEL_NAME}/code_gen.log
wait $!
