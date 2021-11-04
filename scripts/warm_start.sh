SAVE_DIR=models
TB_LOG_DIR=tb_logs/bookcorpus
DATA_NAME=bookcorpus
DATA_PATH=data/${DATA_NAME}.json
MODEL_NAME=discoDVT_warmstart
PRETRAINED_MODEL_DIR=${SAVE_DIR}/bart-base

mkdir -p ${SAVE_DIR}/${MODEL_NAME}

python src/main.py \
--dataset_path ${DATA_PATH} \
--model_checkpoint ${PRETRAINED_MODEL_DIR} \
--save_dir ${SAVE_DIR} \
--tb_log_dir ${TB_LOG_DIR} \
--model_name ${MODEL_NAME} \
--max_src_len 0 \
--max_tgt_len 512 \
--gradient_accumulation_steps 8 \
--num_cnn_layers 3 \
--code_dropout 0.0 \
--alpha 0.1 \
--gumbel_trick \
--train_batch_size 4 \
--valid_batch_size 8 \
--valid_metric loss \
--valid_iterations -1 \
--latent_vocab_size 256 \
--valid_sample_N -1 \
--lr 1e-4 \
--constant_lr \
--n_epochs 1 \
--device cuda:1 \
| tee ${SAVE_DIR}/${MODEL_NAME}/stdout.log
wait $!