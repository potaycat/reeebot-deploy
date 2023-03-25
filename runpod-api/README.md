
```
git clone https://github.com/potaycat/reeebot-deploy.git
cd reeebot-deploy/runpod-api/
git fetch && git pull && docker build .
docker tag 41d3254ba31e longnhfvtap/lucario-sd:1.0
docker push longnhfvtap/lucario-sd:1.0r2
```
