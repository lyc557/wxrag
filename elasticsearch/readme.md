使用 Docker Compose 快速搭建 Elasticsearch + Kibana 环境。

创建 docker-compose.yml：
``` yaml
version: '3.8'
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.12.0
    container_name: elasticsearch
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    ports:
      - "9200:9200"
      - "9300:9300"
    volumes:
      - es_data:/usr/share/elasticsearch/data

  kibana:
    image: docker.elastic.co/kibana/kibana:8.12.0
    container_name: kibana
    depends_on:
      - elasticsearch
    ports:
      - "5601:5601"

volumes:
  es_data:
    driver: local
```

🚀 启动

``` bash
docker-compose up -d
```

✅ 访问

Elasticsearch API: http://192.168.31.120:9200  http://10.45.9.130:9200
配置修改：
``` bash
docker exec -it elasticsearch /bin/bash
```
``` bash
vi config/elasticsearch.yml
```
``` bash
network.host: 0.0.0.0
```
``` bash
exit
```
Kibana 可视化管理: http://192.168.31.120:5601
