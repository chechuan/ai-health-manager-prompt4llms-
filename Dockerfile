FROM registry.enncloud.cn/aimp.en.laikang.com/ai-health-manager-prompt4llms:1.0
WORKDIR ./ai-health-manager-prompt4llms
COPY requirements.txt ./
RUN pip install -r requirements.txt -i https://mirrors.bfsu.edu.cn/pypi/web/simple -U
COPY . .
CMD ["sh","doc/deploy_proj/run_dev.sh"]