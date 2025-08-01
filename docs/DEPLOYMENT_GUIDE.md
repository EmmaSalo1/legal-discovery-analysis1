# Deployment Guide

This guide explains how to make the Legal Discovery Analysis System accessible on the internet for testing across multiple devices.

## 1. Provision a host
- Use any cloud provider (e.g., AWS, DigitalOcean, Render) that supports Docker.
- Open ports 80 or 443 if you plan to serve via HTTP/HTTPS.

## 2. Install dependencies
```bash
sudo apt-get update
sudo apt-get install -y docker docker-compose
```
(Commands may vary based on the host's OS.)

## 3. Clone and configure the project
```bash
git clone <repository-url>
cd legal-discovery-analysis
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY and any other secrets.
# On Render, tesseract is usually located at `/usr/bin/tesseract`, so update
# `TESSERACT_PATH` accordingly.
```

## 4. Start the application
```bash
docker-compose up --build -d
```
This launches the API on port `8000` and the Chroma database on `8001`.

## 5. Accessing the app
Navigate to `http://<server-ip>:8000` in your browser. If you configured a domain
and reverse proxy (e.g., via nginx), you can expose it on standard web ports for
convenience.

Once running, colleagues on different networks can reach the application via the
public address of your server.

