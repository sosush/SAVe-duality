# SAVe-duality

![License](https://img.shields.io/badge/license-MIT-blue.svg)

**SAVe-duality** is a high‑performance data‑streaming and inference platform that captures, processes, and visualises dual‑modality data (e.g., video and audio) in real‑time. It combines a Flask web dashboard, PyTorch models, Apache Iceberg for data lake storage, optional Kafka streaming, and Spark‑based synchronization, providing a seamless end‑to‑end solution for research and production workloads.

---

## Table of Contents

- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Configuration](#configuration)
- [Docker Support](#docker-support)
- [Testing & Linting](#testing--linting)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Features

- **Real‑time video/audio capture** using OpenCV and SoundDevice.
- **GPU‑accelerated inference** with PyTorch and pretrained torchvision models.
- **Web dashboard** built with Flask & Flask‑CORS for live streaming of predictions.
- **Scalable storage** via Apache Iceberg and PyIceberg catalog integration.
- **Optional Kafka hook** for publishing inference results to a Kafka topic.
- **Spark sync engine** for batch processing and downstream analytics.
- **Dockerised deployment** for reproducible environments.

---

## Architecture Overview

```
+-------------------+          +-------------------+          +-------------------+
|  Video & Audio   |  -->     |   Inference       |  -->     |   Flask Dashboard |
|   Capture (cv2)  |          | (PyTorch)         |          |   (Web UI)        |
+-------------------+          +-------------------+          +-------------------+
        |                               |
        |                               v
        |                       +-------------------+
        |                       |   Kafka (opt.)    |
        |                       +-------------------+
        v                               |
+-------------------+                  v
|   Iceberg Store   |          +-------------------+
| (PyIceberg)       |<---------|   Spark Sync      |
+-------------------+          +-------------------+
```

---

## Installation

1. **Clone the repository** (already done) and navigate to the project root.
2. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Install system dependencies** (Ubuntu/Debian example):
   ```bash
   sudo apt-get update && sudo apt-get install -y ffmpeg libsm6 libxext6
   ```
   *On macOS, ensure Homebrew is installed and run:*
   ```bash
   brew install ffmpeg
   ```

---

## Running the Application

Start the Flask dashboard (default runs on `http://127.0.0.1:5000`):
```bash
python web_dashboard.py
```

The dashboard will begin capturing video/audio, run inference, and stream results to the web UI.

### Optional Kafka Producer
If you have a Kafka broker running locally, the application will automatically publish inference results to the `inference-results` topic. Make sure the broker is reachable at `localhost:9092` or edit the connection string in `web_dashboard.py`.

---

## Configuration

All configurable parameters are defined at the top of `web_dashboard.py` and include:
- `KAFKA_PRODUCER` – Kafka connection settings.
- Model selection – modify the torchvision model import if you wish to use a different backbone.
- Iceberg catalog – update the catalog configuration in `streaming/spark_sync_engine.py`.

---

## Docker Support

A Docker Compose file is provided for quick local deployment:
```bash
docker compose up --build
```
This will spin up the Flask app, a Kafka broker, and a Spark container (if configured). Adjust `docker/docker-compose.yml` as needed.

---

## Testing & Linting

The repository includes a basic test suite using `pytest`. Run tests with:
```bash
pytest
```
Lint the codebase with `flake8` or `black` for formatting consistency.

---

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes with clear messages.
4. Push to your fork and open a Pull Request.
5. Ensure all tests pass and adhere to the coding style.

---

## License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

