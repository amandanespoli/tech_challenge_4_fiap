FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Instalar dependências do sistema necessárias para a aplicação
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    python3-dev \
    build-essential \
    ffmpeg \
    libasound2-dev \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Motivo: 
# - portaudio19-dev: Necessário para PyAudio
# - python3-dev, build-essential: Para compilar dependências
# - ffmpeg: Para processamento de vídeo (processar_video_xray)
# - libasound2-dev, libsndfile1: Para sounddevice (microfone do PC)

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copiar código da aplicação
COPY . .

# Criar diretórios necessários com permissões corretas
RUN mkdir -p uploads chromasaude data static templates && \
    chmod 755 uploads chromasaude data static templates

# Motivo: Garante que os diretórios existem e têm permissões adequadas

# Criar usuário não-root
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 5000

# Comando para iniciar a aplicação
CMD ["python", "chatbot.py"]