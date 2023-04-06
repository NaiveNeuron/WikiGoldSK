# Download training script run_ner.py from
# https://github.com/huggingface/transformers/blob/main/examples/pytorch/token-classification/run_ner.py

python3 run_ner.py \
  --model_name_or_path gerulata/slovakbert \
  --train_file ./dev_hg.json \
  --validation_file ./dev_hg.json \
  --test_file ./test_hg.json \
  --text_column_name sentence \
  --label_column_name word_labels \
  --output_dir ./output \
  --overwrite_output_dir \
  --do_train \
  --do_eval \
  --num_train_epochs 10 \
  --evaluation_strategy epoch \
  --save_strategy no \
  --do_predict \
  --max_seq_length 1000 \
  --per_gpu_train_batch_size 32

