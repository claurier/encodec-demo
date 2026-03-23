FROM python:3.13-slim

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download model weights at build time so the app starts instantly
RUN python -c "\
from encodec import EncodecModel; \
EncodecModel.encodec_model_24khz(); \
EncodecModel.encodec_model_48khz()"

COPY encodec_demo.py .

EXPOSE 7860
CMD ["python", "encodec_demo.py"]
