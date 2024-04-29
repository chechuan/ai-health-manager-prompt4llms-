export ENV=test
export LOG_CONSOLE_LEVEL=DEBUG
export LOG_FILE_LEVEL=DEBUG
python src/server.py \
    --port 26921 \
    --special_prompt_version
