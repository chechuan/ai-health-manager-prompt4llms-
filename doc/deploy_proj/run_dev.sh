export ENV=dev
export LOG_CONSOLE_LEVEL=TRACE5
export LOG_FILE_LEVEL=TRACE
python src/server.py \
    --port 6500 \
    --special_prompt_version
    # --use_proxy
