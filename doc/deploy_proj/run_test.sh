export OPENAI_API_KEY=sk-QblVOtsLjhxmmnMUDb9cC741987e4851904e4447BfD38855
export OPENAI_BASE_URL=http://10.228.67.99:26928
python src/server.py \
    --env test \
    --port 26921 \
    --special_prompt_version
