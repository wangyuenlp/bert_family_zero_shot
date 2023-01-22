

MODEL_NAME_OR_PATH=roberta-large
# imdb,ag_news,amazon_polarity,dbpedia_14,yelp_review_full,emotion,yahoo_answers_topics
export CUDA_VISIBLE_DEVICES=4
python train.py \
  --ensemble_method average \
  --similarwords 1\
  --task amazon_polarity \
  --start 1 \
  --end 0 \
  --maskratio 0.05 \
  --model_type roberta \
  --model_name_or_path ${MODEL_NAME_OR_PATH} \
  --max_len 512 \
  --per_gpu_eval_batch_size 64 \
