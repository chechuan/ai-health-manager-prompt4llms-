model=Qwen2-7B-Instruct
concurrency=1
prompt_type=long

python -m src.test.test_llm_throughput \
    --model $model \
    --concurrency $concurrency \
    --prompt-type $prompt_type


