# imdb,ag_news,amazon_polarity,dbpedia_14

MODEL_NAME_OR_PATH=roberta-large

python train.py \
  --ensemble_method average \
  --similarwords 50\
  --task dbpedia_14 \
  --startmasknumber 1 \
  --endmasknumber 1 \
  --model_type roberta \
  --model_name_or_path ${MODEL_NAME_OR_PATH} \
  --max_len 512 \
  --per_gpu_eval_batch_size 64 \
