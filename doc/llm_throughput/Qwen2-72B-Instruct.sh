model=Qwen1.5-32B-Chat
prompt_type=long

python -m src.test.test_llm_throughput \
    --model $model \
    --prompt-type $prompt_type \
    --base-url http://10.228.67.99:26932/v1 \
    --concurrency-lst "1,5,10,20,50" \
    --alpha-times 10
