model=Qwen2-7B-Instruct
concurrency=50
prompt_type=short

python -m src.test.test_llm_throughput \
    --model $model \
    --concurrency $concurrency \
    --prompt-type $prompt_type


