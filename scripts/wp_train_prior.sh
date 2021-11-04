SAVE_DIR=models
TB_LOG_DIR=tb_logs/wp
DATA_NAME=wp
DATA_PATH=data/${DATA_NAME}_code.json
MODEL_NAME=prior-${DATA_NAME}
PRETRAINED_MODEL_DIR=${SAVE_DIR}/bart-base
PRETRAINED_POSTERIOR_DIR=${SAVE_DIR}/discoDVT_${DATA_NAME}

mkdir -p ${SAVE_DIR}/${MODEL_NAME}

python src/prior.py \
--dataset_path ${DATA_PATH} \
--model_checkpoint ${PRETRAINED_MODEL_DIR} \
--pretrained_posterior_path ${PRETRAINED_POSTERIOR_DIR} \
--save_dir ${SAVE_DIR} \
--tb_log_dir ${TB_LOG_DIR} \
--model_name ${MODEL_NAME} \
--max_src_len 64 \
--max_tgt_len 64 \
--gradient_accumulation_steps 8 \
--save_last \
--train_batch_size 128 \
--valid_batch_size 128 \
--latent_vocab_size 256 \
--valid_metric loss \
--valid_iterations -1 \
--valid_sample_N -1 \
--lr 1e-4 \
--warmup_steps 0 \
--warmup_ratio 0.0 \
--weight_decay 0.0 \
--n_epochs 100 \
--device cuda:1 \
| tee ${SAVE_DIR}/${MODEL_NAME}/stdout.log
wait $!