# EnCodec demo 

A Gradio interface for Meta's [EnCodec](https://github.com/facebookresearch/encodec) neural audio codec. Upload an audio file, choose a bitrate, and compare the original against the RVQ reconstruction with side-by-side spectrograms and SNR metrics.

**Models available:**
- EnCodec 24 kHz · mono
- EnCodec 48 kHz · stereo

**Bitrates:** 3.0 / 6.0 / 12.0 / 24.0 kbps

We provide the code to run it locally and/or deploy on a server

---

## Running locally

**Prerequisites:** Docker and Docker Compose installed.

```bash
docker compose up --build
```

The app is then available at [http://localhost](http://localhost).

> First build takes a few minutes — model weights (~200 MB) are downloaded and baked into the image.
> Subsequent builds are fast thanks to Docker layer caching.

---

## Deploying to a server (EC2 or similar)

```bash
# One-time setup on the server
sudo apt update && sudo apt install -y docker.io docker-compose-v2
sudo usermod -aG docker $USER
newgrp docker

# Clone and start
git clone https://github.com/your-username/your-repo.git
cd your-repo
docker compose up -d --build
```

To update after a code change:

```bash
git pull
docker compose up -d --build
```
