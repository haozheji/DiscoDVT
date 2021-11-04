SAVE_DIR=models
DATA_NAME=wp
DATA_PATH=data/${DATA_NAME}_code.json
MODEL_NAME=prior-${DATA_NAME}


python src/prior.py \
--dataset_path ${DATA_PATH} \
--model_checkpoint ${SAVE_DIR}/${MODEL_NAME} \
--save_dir ${SAVE_DIR} \
--model_name ${MODEL_NAME} \
--max_src_len 64 \
--max_tgt_len 64 \
--temperature 1.0 \
--min_length_ratio 0.7 \
--latent_vocab_size 256 \
--valid_batch_size 128 \
--test_sample_N 1000 \
--device cuda:1 \
--do_generate \
| tee ${SAVE_DIR}/${MODEL_NAME}/prior_gen.log
wait $!
