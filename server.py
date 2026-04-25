"""
=============================================================
  Object Detection Streaming Server
  Project #8 — Multi-client TCP Socket + YOLOv8 Inference
=============================================================

Architecture:
  - Main thread listens for incoming TCP connections
  - Each client gets its own handler thread (thread-per-client model)
  - A single shared YOLOv8 model handles inference for all clients
  - Frame protocol: [4-byte size header] + [JPEG bytes]

Author  : Srijita 
Date    : 2026
"""

import socket
import struct
import threading
import time
import cv2
import numpy as np
from ultralytics import YOLO

# ─────────────────────────────────────────────
#  Server Configuration
# ─────────────────────────────────────────────
HOST = "0.0.0.0"          # Listen on all network interfaces
PORT = 9999                # Port clients will connect to
YOLO_MODEL = "yolov8n.pt"  # 'n' = nano (fast); swap to yolov8s.pt for better accuracy
JPEG_QUALITY = 80          # 0–100; lower = smaller payload, faster transfer

# ─────────────────────────────────────────────
#  Global State
# ─────────────────────────────────────────────
# Shared YOLO model — loaded once, reused by all threads.
# ultralytics YOLO is thread-safe for .predict() calls.
model = YOLO(YOLO_MODEL)

# Thread-safe client registry: maps client address → thread name
clients_lock = threading.Lock()
active_clients: dict = {}

# Performance counters (protected by their own lock)
stats_lock = threading.Lock()
total_frames_processed = 0



#  Utility: Send / Receive a framed message


def send_frame(conn: socket.socket, data: bytes) -> None:
    """
    Send bytes over a socket with a 4-byte length prefix.

    Why length-prefix?
    TCP is a stream protocol — it doesn't preserve message boundaries.
    Without a size header, the receiver can't know where one frame ends
    and the next begins.

    Format: [4 bytes: unsigned int = len(data)] [data bytes]
    """
    size = struct.pack(">I", len(data))   # Big-endian unsigned 32-bit int
    conn.sendall(size + data)


def recv_frame(conn: socket.socket) -> bytes | None:
    """
    Receive a length-prefixed frame from a socket.

    Reads exactly 4 bytes first to get the payload size,
    then reads exactly that many bytes.

    Returns None if the connection was closed gracefully.
    """
    # Step 1: Read the 4-byte header
    raw_size = _recv_exact(conn, 4)
    if raw_size is None:
        return None

    payload_size = struct.unpack(">I", raw_size)[0]

    # Sanity check — reject unusually large frames (> 10 MB)
    if payload_size > 10 * 1024 * 1024:
        print(f"[WARN] Suspiciously large frame: {payload_size} bytes — dropping")
        return None

    # Step 2: Read exactly payload_size bytes
    return _recv_exact(conn, payload_size)


def _recv_exact(conn: socket.socket, n: int) -> bytes | None:
    """
    Helper: read exactly `n` bytes from a socket.

    socket.recv() may return fewer bytes than requested
    (TCP segmentation), so we loop until we have enough.
    """
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None   # Connection closed
        buf += chunk
    return buf


# ─────────────────────────────────────────────
#  Core: Client Handler Thread
# ─────────────────────────────────────────────

def handle_client(conn: socket.socket, addr: tuple) -> None:
    """
    Runs in its own thread for each connected client.

    Lifecycle:
      1. Register client
      2. Loop: receive frame → run YOLO → send annotated frame
      3. Deregister client on disconnect or error

    Thread safety:
      - clients_lock protects active_clients dict
      - stats_lock protects total_frames_processed counter
      - YOLO .predict() is thread-safe (ultralytics handles it internally)
    """
    global total_frames_processed

    thread_name = threading.current_thread().name
    print(f"[+] Client connected: {addr}  (thread: {thread_name})")

    # Register this client
    with clients_lock:
        active_clients[addr] = thread_name

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            # ── Step 1: Receive a JPEG frame from the client ──────────────
            raw_data = recv_frame(conn)
            if raw_data is None:
                print(f"[~] Client {addr} disconnected")
                break

            # ── Step 2: Decode JPEG bytes → OpenCV BGR image ──────────────
            np_arr = np.frombuffer(raw_data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                print(f"[WARN] Could not decode frame from {addr}")
                continue

            # ── Step 3: Run YOLOv8 object detection ───────────────────────
            # verbose=False suppresses per-frame console output
            results = model.predict(frame, verbose=False)

            # ── Step 4: Draw bounding boxes + labels on the frame ──────────
            annotated_frame = results[0].plot()

            # ── Step 5: Encode annotated frame back to JPEG ───────────────
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            success, encoded = cv2.imencode(".jpg", annotated_frame, encode_params)

            if not success:
                print(f"[WARN] Could not encode annotated frame for {addr}")
                continue

            # ── Step 6: Send the annotated frame back ─────────────────────
            send_frame(conn, encoded.tobytes())

            # ── Step 7: Update stats ──────────────────────────────────────
            frame_count += 1
            with stats_lock:
                total_frames_processed += 1

            # Print FPS every 30 frames
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"[Stats] {addr} → {frame_count} frames | {fps:.1f} FPS")

    except ConnectionResetError:
        print(f"[!] {addr} — connection reset by client")
    except Exception as e:
        print(f"[ERROR] {addr} — {e}")
    finally:
        # Always clean up, even on unexpected errors
        conn.close()
        with clients_lock:
            active_clients.pop(addr, None)
        print(f"[-] Client {addr} removed. Active clients: {len(active_clients)}")


# ─────────────────────────────────────────────
#  Main: Server Entry Point
# ─────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  Object Detection Streaming Server")
    print(f"  Model : {YOLO_MODEL}")
    print(f"  Host  : {HOST}:{PORT}")
    print("=" * 50)

    # Pre-warm the model with a dummy frame so the first real client
    # doesn't pay the model initialization cost
    print("[*] Warming up YOLO model...")
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    model.predict(dummy, verbose=False)
    print("[*] Model ready.\n")

    # Create the server socket
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # SO_REUSEADDR: allows re-using the port immediately after restart
    # (avoids "Address already in use" during development)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    server_sock.bind((HOST, PORT))
    server_sock.listen(10)   # Queue up to 10 pending connections
    print(f"[*] Listening on {HOST}:{PORT} ...")

    try:
        while True:
            # accept() blocks until a client connects
            conn, addr = server_sock.accept()

            # Spawn a daemon thread for this client.
            # daemon=True means the thread dies automatically when main thread exits.
            t = threading.Thread(
                target=handle_client,
                args=(conn, addr),
                name=f"Client-{addr[1]}",
                daemon=True
            )
            t.start()

            with clients_lock:
                print(f"[*] Active clients: {len(active_clients)}")

    except KeyboardInterrupt:
        print("\n[*] Shutting down server...")
    finally:
        server_sock.close()
        print(f"[*] Total frames processed: {total_frames_processed}")


if __name__ == "__main__":
    main()
