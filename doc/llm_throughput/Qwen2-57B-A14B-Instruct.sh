model=Qwen2-57B-A14B-Instruct
prompt_type=short

python -m src.test.test_llm_throughput \
    --model $model \
    --prompt-type $prompt_type \
    --base-url http://10.228.67.99:26933/v1 \
    --concurrency-lst "1,5,10,20,50" \
    --alpha-times 10
