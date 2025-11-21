# WITNESS - Dockerfile
# For future deployment to Jetson Orin Nano
# Build: docker build -t witness:ghost .
# Run: docker run --rm -it --device /dev/snd witness:ghost

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    alsa-utils \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Piper TTS
RUN pip install piper-tts

# Create app directory
WORKDIR /witness

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY witness_voice.py .

# Download Whisper model on build (saves time at runtime)
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8')"

# Set environment
ENV OLLAMA_HOST=http://host.docker.internal:11434

# Run the witness
CMD ["python", "witness_voice.py"]
