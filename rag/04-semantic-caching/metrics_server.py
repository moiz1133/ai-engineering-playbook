"""Starts a Prometheus /metrics HTTP endpoint on port 8000.

Run this alongside your application:
  python metrics_server.py

Then in Grafana, add data source: http://localhost:8000
Scrape interval: 15s
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prometheus_client import start_http_server

from observability.prometheus_metrics import REGISTRY

if __name__ == "__main__":
    port = 8000
    start_http_server(port, registry=REGISTRY)
    print(f"[PROMETHEUS] Metrics server running at http://localhost:{port}/metrics")
    print("Add as a Prometheus scrape target, then import a dashboard in Grafana")
    print("Ctrl+C to stop")
    while True:
        time.sleep(1)

# WHAT: start_http_server exposes /metrics in Prometheus text format
# WHY: Prometheus scrapes this endpoint every 15s and stores time-series data;
#      Grafana reads from Prometheus to build dashboards
# NOTE: for production, integrate with a FastAPI app using
#       prometheus_client.make_asgi_app() instead of a separate server
