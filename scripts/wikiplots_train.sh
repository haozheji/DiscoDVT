SAVE_DIR=models
TB_LOG_DIR=tb_logs/wikiplots
DATA_NAME=wikiplots
DATA_PATH=data/${DATA_NAME}.json
MODEL_NAME=discoDVT_${DATA_NAME}
PRETRAINED_MODEL_DIR=${SAVE_DIR}/discoDVT_warmstart

mkdir -p ${SAVE_DIR}/${MODEL_NAME}

python src/main.py \
--dataset_path ${DATA_PATH} \
--model_checkpoint ${PRETRAINED_MODEL_DIR} \
--save_dir ${SAVE_DIR} \
--tb_log_dir ${TB_LOG_DIR} \
--model_name ${MODEL_NAME} \
--load_from_full \
--max_src_len 16 \
--max_tgt_len 512 \
--gradient_accumulation_steps 4 \
--num_cnn_layers 3 \
--code_dropout 0.0 \
--alpha 0.1 \
--gamma 0.1 \
--workers 10 \
--train_batch_size 4 \
--valid_batch_size 8 \
--valid_iterations -1 \
--latent_vocab_size 256 \
--gumbel_trick \
--gumbel_anneal \
--gumbel_anneal_rate 1e-4 \
--valid_sample_N -1 \
--lr 1e-4 \
--n_epochs 5 \
--device cuda:0 \
| tee ${SAVE_DIR}/${MODEL_NAME}/stdout.log
wait $!