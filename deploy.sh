#!/bin/bash
# 部署到 Contabo 服务器
# 用法: ./deploy.sh

SERVER="root@161.97.72.212"
REMOTE_DIR="/opt/pencilstroke"

echo "=== Deploying to Contabo ==="

ssh $SERVER "mkdir -p $REMOTE_DIR"
scp collect_server.py $SERVER:$REMOTE_DIR/
scp requirements.txt $SERVER:$REMOTE_DIR/

ssh $SERVER << 'EOF'
cd /opt/pencilstroke
pip3 install -r requirements.txt 2>/dev/null || pip install -r requirements.txt

# 创建 systemd 服务
cat > /etc/systemd/system/pencilstroke.service << 'SERVICE'
[Unit]
Description=PencilStroke Data Collector
After=network.target

[Service]
WorkingDirectory=/opt/pencilstroke
ExecStart=/usr/local/bin/uvicorn collect_server:app --host 0.0.0.0 --port 8400
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
SERVICE

systemctl daemon-reload
systemctl enable pencilstroke
systemctl restart pencilstroke
systemctl status pencilstroke --no-pager

echo ""
echo "=== Done ==="
echo "Test: curl http://161.97.72.212:8400/api/status"
EOF
