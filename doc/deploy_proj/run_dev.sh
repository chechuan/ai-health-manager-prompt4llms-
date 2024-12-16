export ENV=dev
export LOG_CONSOLE_LEVEL=DEBUG
export LOG_FILE_LEVEL=DEBUG
aliyun-instrument python src/server.py \
    --port 6500 \
    --special_prompt_version
    # --use_proxy