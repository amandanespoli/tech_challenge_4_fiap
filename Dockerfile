FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8080

# Instalar dependências do sistema necessárias para a aplicação
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    python3-dev \
    build-essential \
    ffmpeg \
    libasound2-dev \
    libsndfile1 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Motivo das dependências:
# - python3-dev, build-essential: compilação de extensões C (alguns pacotes pip)
# - ffmpeg: processamento de vídeo em gravar_e_transcrever.py

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY . .

# Criar diretórios com permissões corretas
RUN mkdir -p chromasaude data && \
    chmod 755 chromasaude data

# Criar usuário não-root por segurança
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

ENV PORT=8080
# Expor porta 8080 (configurada em config.py)
EXPOSE 8080

# Executar init_container.py antes de iniciar a app
# Valida ambiente e sincroniza ChromaDB do GCS se necessário
CMD ["sh", "-c", "python init_container.py && python chatbot.py"]
