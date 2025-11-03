@echo off
REM Example with publicly available models (no auth required)
REM Options: microsoft/phi-2, microsoft/phi-1_5, google/gemma-2b

python scripts/run_hf_pipeline.py --config configs/hf_example_public.yaml --teacher-model microsoft/phi-2 --student-model microsoft/phi-1_5 --distill-data data/distill.txt --finetune-data data/finetune.txt --seq-len 2048 --stride 1024 --batch-size 1

