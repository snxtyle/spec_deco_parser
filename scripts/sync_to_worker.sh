#!/bin/bash
# Sync code and data to worker node
# Usage: ./sync_to_worker.sh <worker_ip> [worker_user]

WORKER_IP=${1:-"192.168.1.105"}
WORKER_USER=${2:-"suraj"}
PROJECT_PATH="/home/suraj/Desktop/P_Eagle"
REMOTE_PATH="~/P_Eagle"

echo "Syncing to $WORKER_USER@$WORKER_IP..."

# Sync code
rsync -avz --exclude='venv/' --exclude='checkpoints/' --exclude='.git/' \
    $PROJECT_PATH/ $WORKER_USER@$WORKER_IP:$REMOTE_PATH/

# Sync feature data
echo "Syncing features..."
rsync -avz $PROJECT_PATH/data/features/ $WORKER_USER@$WORKER_IP:$REMOTE_PATH/data/features/

echo "Done! Run ./automation.sh multi <master_ip> on both machines."
