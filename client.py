"""
=============================================================
  Object Detection Streaming Client
  Project #8 — Webcam → TCP → Server (YOLO) → Display
=============================================================

How it works:
  1. Connect to the server via TCP
  2. Capture a webcam frame using OpenCV
  3. Compress to JPEG and send with a 4-byte length prefix
  4. Receive the annotated JPEG frame from server
  5. Decode and display in a window
  6. Repeat until 'q' is pressed

Author  : Srijita Choudary
Date    : 2026
"""

import socket
import struct
import time
import cv2
import numpy as np

# ─────────────────────────────────────────────
#  Client Configuration
# ─────────────────────────────────────────────
SERVER_HOST = "127.0.0.1"   # Change to server's IP for remote connection
SERVER_PORT = 9999
JPEG_QUALITY = 80            # Encoding quality before sending (0–100)
WEBCAM_INDEX = 0             # 0 = default webcam; change if you have multiple


# ─────────────────────────────────────────────
#  Utility: same framing protocol as server
# ─────────────────────────────────────────────

def send_frame(conn: socket.socket, data: bytes) -> None:
    """Send bytes with a 4-byte big-endian length prefix."""
    size = struct.pack(">I", len(data))
    conn.sendall(size + data)


def recv_frame(conn: socket.socket) -> bytes | None:
    """Receive a length-prefixed frame. Returns None on disconnect."""
    raw_size = _recv_exact(conn, 4)
    if raw_size is None:
        return None
    payload_size = struct.unpack(">I", raw_size)[0]
    return _recv_exact(conn, payload_size)


def _recv_exact(conn: socket.socket, n: int) -> bytes | None:
    """Read exactly n bytes from socket (handles TCP segmentation)."""
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


# ─────────────────────────────────────────────
#  Main Client Loop
# ─────────────────────────────────────────────

def main():
    print(f"[*] Connecting to {SERVER_HOST}:{SERVER_PORT} ...")

    # Create TCP socket and connect
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((SERVER_HOST, SERVER_PORT))
    except ConnectionRefusedError:
        print("[ERROR] Could not connect. Is the server running?")
        return

    print("[*] Connected! Starting webcam stream. Press 'q' to quit.")

    # Open the webcam
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Could not open webcam at index {WEBCAM_INDEX}")
        sock.close()
        return

    # Optional: set resolution (lower = faster transfer)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # ── Performance tracking ──────────────────────────────────────────
    frame_count = 0
    start_time = time.time()
    latency_sum = 0.0

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

    try:
        while True:
            # ── Step 1: Capture frame from webcam ─────────────────────
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to capture frame — retrying...")
                continue

            # ── Step 2: Compress frame to JPEG ────────────────────────
            success, encoded = cv2.imencode(".jpg", frame, encode_params)
            if not success:
                print("[WARN] Failed to encode frame")
                continue

            # ── Step 3: Send to server ────────────────────────────────
            t_send = time.time()
            send_frame(sock, encoded.tobytes())

            # ── Step 4: Receive annotated frame from server ───────────
            response_data = recv_frame(sock)
            if response_data is None:
                print("[!] Server disconnected")
                break

            round_trip_ms = (time.time() - t_send) * 1000

            # ── Step 5: Decode and display annotated frame ────────────
            np_arr = np.frombuffer(response_data, dtype=np.uint8)
            annotated = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if annotated is None:
                print("[WARN] Could not decode server response")
                continue

            # Overlay FPS + latency info on the displayed frame
            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            latency_sum += round_trip_ms
            avg_latency = latency_sum / frame_count

            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(annotated, f"Latency: {round_trip_ms:.0f}ms (avg: {avg_latency:.0f}ms)",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

            cv2.imshow("Object Detection — Streaming Client", annotated)

            # ── Step 6: Handle keyboard input ────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[*] 'q' pressed — exiting.")
                break

    except KeyboardInterrupt:
        print("\n[*] Interrupted by user.")
    except BrokenPipeError:
        print("[!] Server closed the connection.")
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        sock.close()

        # Print summary stats
        elapsed = time.time() - start_time
        print(f"\n── Session Summary ──────────────────────")
        print(f"  Frames sent   : {frame_count}")
        print(f"  Duration      : {elapsed:.1f}s")
        print(f"  Average FPS   : {frame_count / elapsed:.1f}" if elapsed > 0 else "")
        print(f"  Avg latency   : {latency_sum / frame_count:.1f}ms" if frame_count > 0 else "")
        print(f"─────────────────────────────────────────")


if __name__ == "__main__":
    main()
