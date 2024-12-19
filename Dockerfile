FROM registry.enncloud.cn/aimp.en.laikang.com/ai-health-manager-prompt4llms:1.1
WORKDIR ./ai-health-manager-prompt4llms
COPY requirements.txt ./
RUN pip install -r requirements.txt -i https://repo.huaweicloud.com/repository/pypi/simple -U
#RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && pip config set install.trusted-host mirrors.aliyun.com
#RUN pip3 install -i https://mirrors.aliyun.com/pypi/simple/ aliyun-bootstrap==1.0.3
#ENV ARMS_REGION_ID=cn-beijing
RUN #aliyun-bootstrap -a install
COPY . .
CMD ["sh","doc/deploy_proj/run_dev.sh"]