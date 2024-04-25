export ENV=fat
export LOG_CONSOLE_LEVEL=DEBUG 
export LOG_FILE_LEVEL=DEBUG
python src/server.py \
    --port 6500 \
    --special_prompt_version
