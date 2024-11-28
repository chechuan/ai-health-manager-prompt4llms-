import asyncio
import httpx
from src.utils.Logger import logger

# 配置服务列表
SERVICES = [
    {"name": "ai-health-manager-prompt4llms-dev", "env": "dev", "url": "http://ai-health-manager-prompt4llms-dev.lk-base-dn:6500"},
    {"name": "ai-health-manager-prompt4llms-pro", "prod": "dev", "url": "http://ai-health-manager-prompt4llms-pro.lk-danao-pro:6500"}
]

# 通知函数
async def notify_service_unhealthy(service_name: str, env: str):
    """
    发送通知到 iCome 或其他通知系统
    """
    url = "https://gate.op.laikang.com/health-check-system/ping/upload"  # 通知接口
    payload = {
        "serviceName": service_name,
        "env": env,
        "status": "unhealthy"
    }
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload)
            if response.status_code == 200:
                logger.info(f"通知发送成功: {service_name} ({env})")
            else:
                logger.error(f"通知发送失败: {response.text}")
    except Exception as e:
        logger.error(f"通知异常: {e}")

# 健康检查函数
async def check_service_health(timeout: int = 10):
    """
    检查所有服务的健康状态
    """
    unhealthy_services = []
    for service in SERVICES:
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(service["url"])
                if response.status_code != 200:
                    logger.warning(f"服务异常: {service['name']} ({service['env']})")
                    unhealthy_services.append(service)
        except Exception as e:
            logger.error(f"服务检查失败: {service['name']} ({service['env']}) - {e}")
            unhealthy_services.append(service)
    return unhealthy_services

# 异步监控任务
async def monitor_services(interval: int = 10):
    """
    定时监控所有服务
    """
    while True:
        try:
            logger.info("开始服务健康检查")
            unhealthy_services = await check_service_health()
            for service in unhealthy_services:
                await notify_service_unhealthy(service["name"], service["env"])
        except Exception as e:
            logger.error(f"监控任务异常: {e}")
        await asyncio.sleep(interval)
