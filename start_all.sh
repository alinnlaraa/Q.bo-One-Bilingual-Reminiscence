#!/bin/bash
set -e  # exit on error

cleanup() {
  if [ -n "$BOOT_BLINK_PID" ]; then
    kill "$BOOT_BLINK_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

#Process start indication for user
echo "[QBO] Starting boot blink (blue)..."
python2 /opt/qbo/qbo_blue_blinker.py &
BOOT_BLINK_PID=$!

#ports 

PORTS_TO_FREE=(50007 60007)

#free ports if in use
echo "[LAUNCHER] Freeing ports..."
for port in "${PORTS_TO_FREE[@]}"; do
    sudo fuser -k ${port}/tcp 2>/dev/null || true
done

# Kill any previous processes just in case
echo "[LAUNCHER] Killing leftover processes..."
pkill -f qbo_client_2.py || true
pkill -f qbo_audio_receiver.py || true
pkill -f server_official.py || true
sleep 1

# Wait until the mini PC is reachable
echo "[LAUNCHER] Waiting for mini PC to be reachable..."
until ping -c1 -W1 Linux-PC.local  &>/dev/null; do
    echo "[LAUNCHER] Mini PC not reachable yet, waiting..."
    sleep 1
done
echo "[LAUNCHER] Mini PC is reachable, continuing..."
echo "[QBO] Stopping boot blink..."
kill $BOOT_BLINK_PID 2>/dev/null || true
sleep 0.5


# Start audio receiver on QBO (Python 3)
echo "[QBO] Starting qbo_audio_receiver.py..."
python3 /opt/qbo/qbo_audio_receiver.py &
RECEIVER_PID=$!

# Give it a moment to ensure the server can connect
sleep 2

echo "[QBO] Killing leftover server processes on mini PC..."
ssh ossi@Linux-PC.local "pkill -f server_official.py" || true
sleep 1

# Start server on mini PC via SSH in the virtualenv
echo "[QBO] Starting server_official.py on mini PC..."
ssh -vvv -o BatchMode=yes -o ConnectTimeout=5 -o PasswordAuthentication=no ossi@Linux-PC.local \
"source /home/ossi/vector_ai/venv-vector/bin/activate && nohup python /home/ossi/vector_ai/server_official.py > /home/ossi/vector_ai/server_official.log 2>&1 & echo LAUNCHED"&
SERVER_PID=$!

# Wait till server listens on port
echo "[LAUNCHER] Waiting for mini PC to listen on port 50007"
until ssh ossi@Linux-PC.local "ss -lnt | grep -q ':50007 '"; do
  echo "[LAUNCHER] Port 50007 not listening yet, wiating..."
  sleep 1
done
echo "[LAUNCHER] Port 50007 is listening."
sleep 1
# Give server time to start and listen for QBO client

# Start QBO client (Python 2)
echo "[QBO] Starting qbo_client_2.py..."
python2 /opt/qbo/qbo_client_2.py &
CLIENT_PID=$!

echo "[QBO] All processes started."
echo "Receiver PID: $RECEIVER_PID"
echo "Server PID: $SERVER_PID"
echo "Client PID: $CLIENT_PID"

# Optional: wait for all to finish
wait $RECEIVER_PID $SERVER_PID $CLIENT_PID
