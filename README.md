# CN_project
# Object Detection Streaming Server — Project #8

## What This Does
- Clients stream live webcam frames over TCP to a central server
- Server detects objects in each frame using YOLOv8 (in real time)
- Annotated frames (with bounding boxes + labels) are sent back
- Multiple clients are handled simultaneously using threads

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```
YOLOv8 model (`yolov8n.pt`) auto-downloads on first run (~6 MB).

### 2. Run the server
```bash
python server.py
```

### 3. Run one or more clients (each in a separate terminal)
```bash
python client.py
```
To connect from another machine, edit `SERVER_HOST` in `client.py` to the server's IP.

---

## Frame Protocol (Custom Binary Protocol)
```
┌─────────────────┬───────────────────────────────┐
│  4 bytes        │  N bytes                      │
│  (payload size) │  (JPEG-encoded frame data)    │
└─────────────────┴───────────────────────────────┘
```
- Size field: big-endian unsigned 32-bit integer (`struct.pack(">I", n)`)
- Payload: JPEG-compressed frame (OpenCV `imencode`)
- Same format in both directions (client→server and server→client)

---

## Thread Handling
```
Main Thread
    └── server_sock.accept()   ← blocks, waiting for clients
         │
         ├── Client 1 → Thread "Client-XXXX"  (handle_client)
         ├── Client 2 → Thread "Client-YYYY"  (handle_client)
         └── Client N → Thread "Client-ZZZZ"  (handle_client)
```
- **One thread per client** — each runs `handle_client()` independently
- **Shared resource (YOLO model)**: ultralytics handles internal thread safety
- **Shared resource (active_clients dict)**: protected by `clients_lock` (threading.Lock)
- **Shared resource (frame counter)**: protected by `stats_lock`
- Threads are **daemon threads** — auto-killed when main thread exits

---

## Performance Analysis Points (for your report)

| Metric            | How to measure                          |
|-------------------|-----------------------------------------|
| FPS (client-side) | Displayed on live video window          |
| Round-trip latency| Printed per-frame in client terminal    |
| CPU usage         | `htop` or Task Manager during demo      |
| Memory usage      | `psutil` or Task Manager                |
| Scalability       | Run 3–5 clients simultaneously          |

### Bottlenecks to discuss:
- **YOLO inference** is the main bottleneck (~30–80ms per frame on CPU)
- **JPEG quality** setting trades image fidelity vs. network bandwidth
- **Thread-per-client** works well up to ~10 clients; beyond that, a thread pool would be better

---

