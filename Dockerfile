FROM registry.enncloud.cn/aimp.en.laikang.com/ai-health-manager-prompt4llms:1.1

WORKDIR ./ai-health-manager-prompt4llms

# 安装依赖
COPY requirements.txt ./
RUN pip install -r requirements.txt -i https://repo.huaweicloud.com/repository/pypi/simple -U
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && pip config set install.trusted-host mirrors.aliyun.com
RUN pip3 install -i https://mirrors.aliyun.com/pypi/simple/ aliyun-bootstrap==1.0.3

# 设置环境变量
ENV ARMS_REGION_ID=cn-beijing
ENV LANGFUSE_SECRET_KEY=sk-lf-dd72d122-597f-4816-be79-605ecdabc575
ENV LANGFUSE_PUBLIC_KEY=pk-lf-006f983d-9437-4d7b-8b01-b17b03a5ab91
ENV LANGFUSE_HOST=https://ai-health-manager-langfuse-web.op.laikang.com

# 安装 aliyun-bootstrap
RUN aliyun-bootstrap -a install

COPY . .

# 启动容器时执行的命令
CMD ["sh", "doc/deploy_proj/run_dev.sh"]
