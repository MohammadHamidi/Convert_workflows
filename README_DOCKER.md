# Docker Deployment Guide

## Quick Start

### 1. Build and Run with Docker Compose

```bash
# Build and start the container
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

### 2. Environment Variables

Create a `.env` file in the project root (optional):

```env
AVALAI_API_KEY=your_api_key_here
PORT=8000
```

Or set environment variables directly:

```bash
export AVALAI_API_KEY=your_api_key_here
docker-compose up -d
```

### 3. Access the Application

Once running, access the application at:
- Frontend: http://localhost:8000
- API Docs: http://localhost:8000/doc
- Health Check: http://localhost:8000/health

## Docker Commands

### Build Only
```bash
docker-compose build
```

### Rebuild (no cache)
```bash
docker-compose build --no-cache
```

### View Logs
```bash
docker-compose logs -f n8n-converter
```

### Stop and Remove
```bash
docker-compose down
```

### Stop, Remove, and Remove Volumes
```bash
docker-compose down -v
```

### Restart Service
```bash
docker-compose restart n8n-converter
```

## Development Mode

For development with hot-reload, uncomment the static volume mount in `docker-compose.yml`:

```yaml
volumes:
  - ./config.json:/app/config.json:ro
  - ./static:/app/static:ro  # Uncomment this line
```

Then rebuild:
```bash
docker-compose up -d --build
```

## Production Deployment

### 1. Build Production Image
```bash
docker-compose build
```

### 2. Run in Production
```bash
docker-compose up -d
```

### 3. Check Health
```bash
curl http://localhost:8000/health
```

## Troubleshooting

### Container won't start
```bash
# Check logs
docker-compose logs n8n-converter

# Check if port is already in use
netstat -an | grep 8000
```

### Rebuild after code changes
```bash
docker-compose up -d --build
```

### Access container shell
```bash
docker-compose exec n8n-converter /bin/bash
```

### Check container status
```bash
docker-compose ps
```

## Docker Image Details

- **Base Image**: Python 3.11-slim
- **Working Directory**: /app
- **Exposed Port**: 8000
- **Health Check**: /health endpoint
- **Restart Policy**: unless-stopped

## Volume Mounts

- `config.json`: Read-only mount for configuration (can be modified without rebuild)
- `static/`: Optional mount for development (uncomment in docker-compose.yml)

## Network

The service runs on a bridge network `converter-network` for potential future service expansion.

