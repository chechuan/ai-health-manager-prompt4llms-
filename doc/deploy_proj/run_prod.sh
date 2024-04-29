export ENV=prod
export LOG_CONSOLE_LEVEL=DEBUG 
export LOG_FILE_LEVEL=DEBUG
python src/server.py \
    --port 26922 \
    --special_prompt_version