#qbo_client_2.py

#!/usr/bin/env python2
# -*- coding: latin-1 -*-

import socket
import subprocess
import sys
import time
import serial
import threading

from controller.QboController import Controller

SERVER_IP = "Linux-PC.local"  # Mini PC IP
PORT = 50007

# -------------------------------------------------
# SERIAL CONFIG (same as your test script)
# -------------------------------------------------
SERIAL_PORT = "/dev/serial0"
BAUDRATE = 115200

# -------------------------------------------------
# NOSE STATES (QBO firmware)
# -------------------------------------------------
NOSE_OFF       = 0x00
NOSE_GREEN = 0x04  # green
NOSE_BLUE  = 0x01  # blue
#NOSE_THINKING  = 0x02  # red

# -------------------------------------------------
# COMMAND BYTES FROM SERVER
# -------------------------------------------------
CMD_SPEAKING = '\x00'   # speaking (blue)
CMD_LISTENING = '\x01'  # listening (green)
CMD_STOP_CONVERSATION = '\x02'   # shutdown
CMD_THINKING = '\x10'   # blinking blue
CMD_OFF = '\x03' #LEDs off
#--------------------------
# BLINKING LOGIC
#--------------------------

blinking = False
blink_thread = None

def blink_blue(qbo):
    global blinking
    while blinking:
        qbo.SetNoseColor(NOSE_BLUE)
        time.sleep(0.4)
        qbo.SetNoseColor(NOSE_OFF)
        time.sleep(0.4)

def stop_blinking():
    global blinking
    blinking = False
    time.sleep(0.1)


def main():
    global blinking, blink_thread

    print("[QBO] Initializing serial and LED controller...")

    ser = serial.Serial(
        SERIAL_PORT,
        BAUDRATE,
        timeout=1
    )
    qbo = Controller(ser)

    # Start with LED off
    qbo.SetNoseColor(NOSE_OFF)

    print("[QBO] Connecting to server {}:{} ...".format(SERVER_IP, PORT))
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((SERVER_IP, PORT))
    except Exception as e:
        print("[QBO] Failed to connect to server:", e)
        ser.close()
        return

    print("[QBO] Connected to server. Waiting for commands...")

    try:
        while True:
            cmd = sock.recv(1)
            if not cmd:
                print("[QBO] Server closed connection.")
                break

            if cmd == CMD_SPEAKING:
                stop_blinking()
                print("[LED] SPEAKING (BLUE)")
                qbo.SetNoseColor(NOSE_BLUE)

            elif cmd == CMD_LISTENING:
                stop_blinking()
                print("[LED] LISTENING (GREEN)")
                qbo.SetNoseColor(NOSE_GREEN)

            elif cmd == CMD_THINKING:
                if not blinking:
                    print("[LED] THINKING (BLINKING BLUE)")
                    blinking = True
                    blink_thread = threading.Thread(target=blink_blue, args=(qbo,))
                    blink_thread.daemon = True
                    blink_thread.start()

#                qbo.SetNoseColor(NOSE_THINKING)


            elif cmd == CMD_STOP_CONVERSATION:
                print("[QBO] SHUTDOWN command received.")
                stop_blinking()
                try:
                    qbo.SetNoseColor(NOSE_OFF)
                except:
                    pass
                ser.close()
                sock.close()
                # Small delay to flush prints
                time.sleep(0.5)
                # Shutdown QBO
                subprocess.Popen(["sudo", "poweroff"])
                sys.exit(0)

            elif cmd == CMD_OFF:
                print("[LED] STOPWORD received: turning LEDs off.")
                stop_blinking()
                qbo.SetNoseColor(NOSE_OFF)
                ser.close()
                sock.close()
                sys.exit(0)

            else:
                print("[QBO] Unknown command byte:", repr(cmd))

    except KeyboardInterrupt:
        print("\n[QBO] Interrupted by user.")

    finally:
        stop_blinking()
        try:
            qbo.SetNoseColor(NOSE_OFF)
        except:
            pass
        ser.close()
        sock.close()
        print("[QBO] Serial and socket closed.")

if __name__ == "__main__":
    main()