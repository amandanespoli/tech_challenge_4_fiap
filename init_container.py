#!/usr/bin/env python3
"""
Script de inicialização do container Cloud Run.
Valida variáveis de ambiente e sincroniza ChromaDB do GCS.
"""

import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


def main():
    """Inicialização do container"""
    try:
        logger.info("=" * 70)
        logger.info("🚀 Inicializando Container Cloud Run...")
        logger.info("=" * 70)
        
        # 1. Validar API Key (OBRIGATÓRIO)
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.error("❌ OPENAI_API_KEY não configurada!")
            logger.error("   Adicione em Secret Manager: gcloud secrets create OPENAI_API_KEY --data-file=-")
            return False
        logger.info("✅ OPENAI_API_KEY carregada")
        
        # 2. Detectar se é Cloud Run
        is_cloud_run = os.getenv('K_SERVICE') is not None
        if is_cloud_run:
            logger.info("✅ Executando em Cloud Run")
            
            # 3a. Sincronizar Modelos do GCS
            logger.info("")
            from sync_models import sync_models_startup
            sync_models_startup()
            
            # 3b. Sincronizar ChromaDB do GCS
            logger.info("")
            from sync_chromadb import sync_chromadb_startup, get_gcs_bucket
            
            bucket = get_gcs_bucket()
            if not bucket:
                logger.warning("⚠️  GCS_BUCKET não configurada - ChromaDB não será sincronizado")
            else:
                sync_chromadb_startup()
        else:
            logger.info("ℹ️  Executando localmente (não em Cloud Run)")
        
        # 4. Validar que chromasaude/ existe
        logger.info("")
        from config import CHROMA_PATH, MODEL_PATH
        
        # Verificar modelo
        if MODEL_PATH.exists():
            model_size = MODEL_PATH.stat().st_size
            if model_size > 1_000_000:  # > 1MB
                logger.info(f"✅ Modelo de X-ray disponível ({model_size / 1_000_000:.1f}MB)")
            else:
                logger.warning(f"⚠️  {MODEL_PATH} muito pequeno ({model_size} bytes) - pode ser ponteiro Git LFS")
        else:
            logger.warning(f"⚠️  {MODEL_PATH} não existe")
            if not is_cloud_run:
                logger.info("   Execute: git lfs install && git lfs pull")
        
        # Verificar ChromaDB
        if CHROMA_PATH.exists():
            chroma_files = list(CHROMA_PATH.rglob('*'))
            if chroma_files:
                logger.info(f"✅ ChromaDB disponível com {len([f for f in chroma_files if f.is_file()])} arquivo(s)")
            else:
                logger.warning(f"⚠️  {CHROMA_PATH} vazio - RAG terá contexto limitado")
        else:
            logger.warning(f"⚠️  {CHROMA_PATH} não existe")
            logger.info("   Execute: python create_db.py (com PDFs em data/)")
        
        # 5. Pronto para iniciar app
        logger.info("")
        logger.info("=" * 70)
        logger.info("✅ Container inicializado com sucesso!")
        logger.info("=" * 70)
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro na inicialização: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

