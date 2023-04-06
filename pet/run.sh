python3 -u cli.py \
--method pet \
--pattern_ids 0 1\
--data_dir data/ner \
--eval_set test \
--model_type roberta \
--model_name_or_path gerulata/slovakbert \
--task_name ner \
--output_dir output/ner \
--do_train \
--do_eval \
--pet_per_gpu_train_batch_size 4 \
--pet_gradient_accumulation_steps 4 \
--pet_max_steps 250 \
--sc_per_gpu_unlabeled_batch_size 4 \
--sc_gradient_accumulation_steps 4 \
--sc_max_steps 5000 \
--cache_dir cache 2>&1 | tee ner.log

