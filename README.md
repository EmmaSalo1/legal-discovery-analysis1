# Legal Discovery Analysis System

An AI-powered system for analyzing opposing counsel discovery productions, designed to help legal teams efficiently review large volumes of documents and identify key evidence, contradictions, and strategic opportunities.

## 🎯 Features

### Core Capabilities
- **Multi-format Document Processing**: Handle PDFs, Word docs, emails, Excel files, and more
- **AI-Powered Analysis**: Automated evidence identification and contradiction detection
- **Interactive Chat Interface**: Ask questions about case documents using natural language
- **Timeline Visualization**: Chronological reconstruction of case events
- **Privilege Protection**: Identify and isolate attorney-client privileged communications
- **Case Management**: Multi-case support with isolated data storage

### Document Analysis
- **Evidence Analyzer**: Automatically flag key evidence and assess relevance
- **Contradiction Detector**: Find inconsistencies in witness statements and documents
- **Timeline Builder**: Build comprehensive chronological timelines
- **Entity Tracker**: Track people and organizations across documents
- **Privilege Scanner**: Identify inadvertently produced privileged documents

### Specialized Analyzers
- **Deposition Analysis**: Transcript analysis and witness credibility assessment
- **Email Analysis**: Thread reconstruction and privilege review
- **Contract Analysis**: Key terms identification and breach evidence
- **Financial Analysis**: Financial document review and analysis
- **Expert Report Analysis**: Expert witness report evaluation

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL 14+
- Redis 6+
- Docker & Docker Compose (optional)

### Installation

#### Option 1: Docker (Recommended)
```bash
git clone <repository-url>
cd legal-discovery-analysis
cp .env.example .env
# Edit .env with your configuration
docker-compose up --build
```

### Render Deployment Notes
- Set `TESSERACT_PATH` to `/usr/bin/tesseract` on Render or other Linux hosts.
- **Install required system packages on Debian-based hosts:**
  - `libgl1` (for OpenCV)
  - `ffmpeg` (for video/audio processing)
  - `tesseract-ocr` (for OCR)
  - Example:
    ```bash
    apt-get update && apt-get install -y libgl1 ffmpeg tesseract-ocr
    ```
    Include these in your custom Docker build or install them on the host before running the containers.
- Run the Celery worker alongside the API:
  ```bash
  docker-compose up worker
  ```

#### Option 2: Deploy to a remote host

To make the application accessible from any network, you can run it on a small
cloud server or a Platform-as-a-Service provider that supports Docker. Clone the
repository on the server, configure the environment variables in `.env`, and run
`docker-compose up --build -d`. See [docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md)
for a step-by-step deployment walkthrough.
