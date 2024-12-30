from langfuse import Langfuse

langfuse_client = Langfuse(
    secret_key="your-secret-key",
    public_key="your-public-key",
    host="http://your-langfuse-host"
)

trace = langfuse_client.trace(
    name="test_trace",
    user_id="test_user",
    release="v1.0.0"
)

# 检查是否包含 log_event 和 end 方法
print(dir(trace))
