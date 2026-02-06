# PPIO Sandbox Templates for R2E-Gym Training

This directory contains Dockerfile templates for building PPIO sandbox images for R2E-Gym training repositories.

## Prerequisites

1. Install PPIO CLI:
   ```bash
   npm i -g ppio-sandbox-cli
   ```

2. Set environment variables:
   ```bash
   export PPIO_API_KEY=sk_xxxxx
   export PPIO_ACCESS_TOKEN=$PPIO_API_KEY
   ```

3. Ensure Docker is running:
   ```bash
   docker info
   ```

## Available Templates

| Template | Repository | Samples | Priority |
|----------|-----------|---------|----------|
| r2e-pandas | pandas-dev/pandas | 1,444 | High |
| r2e-numpy | numpy/numpy | 781 | High |
| r2e-pillow | python-pillow/Pillow | 620 | High |
| r2e-orange3 | biolab/orange3 | 482 | Medium |
| r2e-aiohttp | aio-libs/aiohttp | 299 | Medium |
| r2e-tornado | tornadoweb/tornado | 261 | Medium |
| r2e-scrapy | scrapy/scrapy | 215 | Medium |
| r2e-pyramid | Pylons/pyramid | 189 | Low |
| r2e-datalad | datalad/datalad | 179 | Low |
| r2e-coveragepy | nedbat/coveragepy | 108 | Low |

## Building Templates

### Build All Templates

```bash
./build_r2e_templates.sh all
```

### Build Single Template

```bash
./build_r2e_templates.sh pandas
./build_r2e_templates.sh numpy
```

### Build with Proxy (China Network)

```bash
export HTTP_PROXY=http://127.0.0.1:1083
export HTTPS_PROXY=http://127.0.0.1:1083
./build_r2e_templates.sh all
```

## After Building

1. List templates to get IDs:
   ```bash
   ppio-sandbox-cli tpl list
   ```

2. Update `REPO_TEMPLATE_MAP` in `rllm/environments/swe_ppio/ppio_reward.py`:
   ```python
   REPO_TEMPLATE_MAP = {
       # Existing SWE-Bench templates...

       # R2E-Gym templates (add template IDs from step 1)
       "pandas": "r2e-pandas-TEMPLATE_ID",
       "numpy": "r2e-numpy-TEMPLATE_ID",
       "pillow": "r2e-pillow-TEMPLATE_ID",
       "orange3": "r2e-orange3-TEMPLATE_ID",
       "aiohttp": "r2e-aiohttp-TEMPLATE_ID",
       "tornado": "r2e-tornado-TEMPLATE_ID",
       "scrapy": "r2e-scrapy-TEMPLATE_ID",
       "pyramid": "r2e-pyramid-TEMPLATE_ID",
       "datalad": "r2e-datalad-TEMPLATE_ID",
       "coveragepy": "r2e-coveragepy-TEMPLATE_ID",
   }
   ```

## Build Time Expectations

Each template takes approximately 10-15 minutes to build:
1. Pull Docker image
2. Build layers
3. Upload to PPIO
4. Convert to Firecracker microVM

Total for all 10 templates: ~2-3 hours

## Troubleshooting

### Docker BuildKit Issues

```bash
export DOCKER_BUILDKIT=0
./build_r2e_templates.sh pandas
```

### Rate Limit

Wait a few minutes and retry if you see "Rate limit exceeded".

### Network Issues

Configure Docker registry mirrors in `/etc/docker/daemon.json`:
```json
{
    "registry-mirrors": [
        "https://dockerpull.org",
        "https://dockerhub.icu"
    ]
}
```

Then restart Docker:
```bash
sudo systemctl restart docker
```
