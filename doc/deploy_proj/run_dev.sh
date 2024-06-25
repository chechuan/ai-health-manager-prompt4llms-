export ENV=local
export LOG_CONSOLE_LEVEL=TRACE
export LOG_FILE_LEVEL=TRACE
python src/server.py \
    --port 6500 \
    --special_prompt_version
    # --use_proxy
