MODEL_NAME_OR_PATH=roberta-large
# imdb,ag_news,amazon_polarity,dbpedia_14,yelp_review_full,emotion,yahoo_answers_topics
export CUDA_VISIBLE_DEVICES=3
python implict.py \
  --ensemble_method average \
  --similarwords 1\
  --task imdb \
  --prediction_position 0 \
  --startmasknumber 0 \
  --endmasknumber 3 \
  --model_type roberta \
  --model_name_or_path ${MODEL_NAME_OR_PATH} \
  --max_len 512 \
  --per_gpu_eval_batch_size 64 \
