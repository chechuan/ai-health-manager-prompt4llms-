
```sh
docker commit -a "songhaoyanga" -m "处理opencv" dbbf0e38b229 ai-health-manager-prompt4llms:1.1
docker tag ai-health-manager-prompt4llms:1.1 registry.enncloud.cn/aimp.en.laikang.com/ai-health-manager-prompt4llms:1.1
docker push registry.enncloud.cn/aimp.en.laikang.com/ai-health-manager-prompt4llms:1.1
```