@echo off
REM Example: Compress Llama-3-12B using hybrid LRA + KV-Distill + BLT
python scripts/run_hf_pipeline.py --config configs/hf_12b.yaml --teacher-model meta-llama/Meta-Llama-3-12B --student-model meta-llama/Meta-Llama-3-8B-Instruct --distill-data /path/to/distill.txt --finetune-data /path/to/finetune.txt --seq-len 2048 --stride 1024 --batch-size 1

