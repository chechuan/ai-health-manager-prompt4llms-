export ENV=dev
export LOG_CONSOLE_LEVEL=TRACE5
export LOG_FILE_LEVEL=TRACE

export OTEL_SERVICE_NAME=ai-health-manager-prompt4llms-dev
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://tracing-analysis-dc-bj-internal.aliyuncs.com/adapt_da71k5hxwf@a2a96ea73919f75_da71k5hxwf@53df7ad2afe8301/api/otlp/traces
export OTEL_EXPORTER_OTLP_METRICS_ENDPOINT=http://tracing-analysis-dc-bj-internal.aliyuncs.com/adapt_da71k5hxwf@a2a96ea73919f75_da71k5hxwf@53df7ad2afe8301/api/otlp/metrics

opentelemetry-instrument python src/server.py \
    --port 6500 \
    --special_prompt_version
    # --use_proxy
