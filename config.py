"""
Configurações Centralizadas - Cloud-Ready
==========================================
Detecta automaticamente se está rodando em container/cloud e ajusta configurações
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ================================================================================
# DETECÇÃO DE AMBIENTE
# ================================================================================

# Detectar se está rodando em Docker
IS_DOCKER = os.path.exists('/.dockerenv')

# Detectar se está rodando em Google Cloud Run
IS_CLOUD_RUN = os.getenv('K_SERVICE') is not None or os.getenv('IS_CLOUD_RUN') == 'true'

# Detectar ambiente (development, staging, production)
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
IS_PRODUCTION = ENVIRONMENT == 'production'
IS_DEVELOPMENT = ENVIRONMENT == 'development'

# ================================================================================
# PATHS - ADAPTADOS PARA CLOUD
# ================================================================================

BASE_DIR = Path(__file__).parent

if IS_DOCKER:
    # Paths para container/cloud
    MODEL_PATH = Path("/app/Departamento_Medico/melhor_modelo.keras")
    CHROMA_PATH = Path("/app/chromasaude")
    UPLOAD_FOLDER = Path("/tmp/uploads")
    DATA_PATH = Path("/app/data")
    TEMPLATES_FOLDER = Path("/app/templates")
    STATIC_FOLDER = Path("/app/static")
else:
    # Paths para desenvolvimento local
    MODEL_PATH = BASE_DIR / "Departamento_Medico" / "melhor_modelo.keras"
    CHROMA_PATH = BASE_DIR / "chromasaude"
    UPLOAD_FOLDER = BASE_DIR / "uploads"
    DATA_PATH = BASE_DIR / "data"
    TEMPLATES_FOLDER = BASE_DIR / "templates"
    STATIC_FOLDER = BASE_DIR / "static"

# Criar diretórios necessários
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
CHROMA_PATH.mkdir(parents=True, exist_ok=True)

# ================================================================================
# FEATURES DISPONÍVEIS - CLOUD-AWARE
# ================================================================================

FEATURES = {
    # Funcionalidades que NÃO funcionam em container/cloud
    'screen_capture': not IS_DOCKER,      # ❌ Servidor não tem monitor
    'pyaudio_recording': not IS_DOCKER,   # ❌ Servidor não tem microfone
    'video_display': not IS_DOCKER,       # ❌ Servidor não tem display
    'window_control': not IS_DOCKER,      # ❌ Servidor não tem janelas
    
    # Funcionalidades que SEMPRE funcionam
    'xray_upload': True,                  # ✅ Upload de imagens
    'video_processing': True,             # ✅ Processamento sem display
    'text_chat': True,                    # ✅ Chat de texto
    'web_audio_recording': True,          # ✅ Gravação via WebRTC
    'tts': True,                          # ✅ Text-to-Speech
}

# ================================================================================
# API KEYS E SECRETS
# ================================================================================

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError(
        "❌ OPENAI_API_KEY não encontrada!\n"
        "Configure a variável de ambiente OPENAI_API_KEY no arquivo .env"
    )

# Secret key para Flask sessions (gerar uma nova em produção!)
SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-CHANGE-IN-PRODUCTION')
if IS_PRODUCTION and SECRET_KEY == 'dev-secret-key-CHANGE-IN-PRODUCTION':
    raise ValueError("❌ Configure SECRET_KEY em produção!")

# ================================================================================
# FLASK CONFIGURAÇÕES
# ================================================================================

# Bind em 0.0.0.0 para aceitar conexões externas no container
FLASK_HOST = '0.0.0.0' if IS_DOCKER else '127.0.0.1'
# Porto padrão 8080 (cloud-friendly), permitir override via PORT env var
FLASK_PORT = int(os.getenv('PORT', 8080))
FLASK_DEBUG = IS_DEVELOPMENT and not IS_DOCKER

# ================================================================================
# UPLOAD CONFIGURAÇÕES
# ================================================================================

# Tamanho máximo de upload (importante para cloud!)
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'webm'}
ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'ogg', 'webm', 'm4a'}

# ================================================================================
# ÁUDIO CONFIGURAÇÕES
# ================================================================================

AUDIO_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_CHUNK = 1024
RECORDING_TIMEOUT_SEC = 30

# ================================================================================
# MODELO ML CONFIGURAÇÕES
# ================================================================================

IMAGE_SIZE = (256, 256)
CLASS_LABELS = {
    0: 'Covid-19',
    1: 'Normal',
    2: 'Pneumonia Viral',
    3: 'Pneumonia Bacteriana'
}

# ================================================================================
# CHROMADB / RAG CONFIGURAÇÕES
# ================================================================================

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 500
SIMILARITY_THRESHOLD = 0.3
MAX_RESULTS = 5

# Persistência do ChromaDB
CHROMA_PERSIST = True

# ================================================================================
# LOGGING CONFIGURAÇÕES
# ================================================================================

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO' if IS_PRODUCTION else 'DEBUG')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Logging estruturado para cloud (JSON)
USE_JSON_LOGGING = IS_PRODUCTION or os.getenv('USE_JSON_LOGGING', 'false').lower() == 'true'

# ================================================================================
# GUNICORN CONFIGURAÇÕES (PRODUÇÃO)
# ================================================================================

if IS_PRODUCTION:
    GUNICORN_WORKERS = int(os.getenv('GUNICORN_WORKERS', 2))
    GUNICORN_THREADS = int(os.getenv('GUNICORN_THREADS', 4))
    GUNICORN_TIMEOUT = int(os.getenv('GUNICORN_TIMEOUT', 300))  # 5 min para vídeos
    GUNICORN_KEEPALIVE = int(os.getenv('GUNICORN_KEEPALIVE', 5))

# ================================================================================
# HEALTH CHECK
# ================================================================================

HEALTH_CHECK_ENABLED = True

# ================================================================================
# CORS CONFIGURAÇÕES
# ================================================================================

# Habilitar CORS se necessário
ENABLE_CORS = os.getenv('ENABLE_CORS', 'false').lower() == 'true'
CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*').split(',')

# ================================================================================
# CACHE CONFIGURAÇÕES
# ================================================================================

# Cache de embeddings (útil para reduzir custos)
ENABLE_EMBEDDING_CACHE = os.getenv('ENABLE_EMBEDDING_CACHE', 'true').lower() == 'true'
CACHE_TTL_SECONDS = int(os.getenv('CACHE_TTL_SECONDS', 3600))  # 1 hora

# ================================================================================
# RATE LIMITING
# ================================================================================

ENABLE_RATE_LIMITING = IS_PRODUCTION
RATE_LIMIT_PER_MINUTE = int(os.getenv('RATE_LIMIT_PER_MINUTE', 60))

# ================================================================================
# FUNCÕES AUXILIARES
# ================================================================================

def get_feature_status() -> dict:
    """Retorna status de todas as features"""
    return {
        'environment': ENVIRONMENT,
        'is_docker': IS_DOCKER,
        'is_production': IS_PRODUCTION,
        'features': FEATURES,
        'paths': {
            'model': str(MODEL_PATH),
            'chroma': str(CHROMA_PATH),
            'uploads': str(UPLOAD_FOLDER),
        }
    }

def is_feature_enabled(feature_name: str) -> bool:
    """Verifica se uma feature está habilitada"""
    return FEATURES.get(feature_name, False)

def get_allowed_extensions(file_type: str) -> set:
    """Retorna extensões permitidas por tipo"""
    mapping = {
        'image': ALLOWED_IMAGE_EXTENSIONS,
        'video': ALLOWED_VIDEO_EXTENSIONS,
        'audio': ALLOWED_AUDIO_EXTENSIONS,
    }
    return mapping.get(file_type, set())

# ================================================================================
# VALIDAÇÃO DE CONFIGURAÇÃO
# ================================================================================

def validate_config():
    """Valida se todas as configurações necessárias estão presentes"""
    errors = []
    
    # Verificar modelo
    if not MODEL_PATH.exists():
        errors.append(f"❌ Modelo não encontrado: {MODEL_PATH}")
    
    # Verificar diretórios
    if not UPLOAD_FOLDER.exists():
        errors.append(f"❌ Pasta de uploads não existe: {UPLOAD_FOLDER}")
    
    # Verificar API key
    if not OPENAI_API_KEY:
        errors.append("❌ OPENAI_API_KEY não configurada")
    
    if errors:
        for error in errors:
            print(error)
        raise RuntimeError("Configuração inválida! Corrija os erros acima.")
    
    print("✅ Configuração validada com sucesso!")
    print(f"   Ambiente: {ENVIRONMENT}")
    print(f"   Docker: {IS_DOCKER}")
    print(f"   Features disponíveis: {sum(FEATURES.values())}/{len(FEATURES)}")

# ================================================================================
# EXIBIR CONFIGURAÇÃO (DEBUG)
# ================================================================================

if __name__ == "__main__":
    import json
    print("=" * 80)
    print("CONFIGURAÇÕES DO SISTEMA")
    print("=" * 80)
    print(json.dumps(get_feature_status(), indent=2))
    print("\n" + "=" * 80)
    validate_config()