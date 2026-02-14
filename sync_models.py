#!/usr/bin/env python3
"""
Módulo de sincronização de Modelos de ML com Google Cloud Storage.
Baixa modelos treinados (ResNet50 para X-ray) do GCS apenas em Cloud Run.

Localmente: usa Git LFS (install git-lfs && git lfs pull)
Cloud Run: baixa automaticamente do GCS no startup
"""
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def is_cloud_run() -> bool:
    """Detecta se está rodando em Cloud Run (Google Cloud)."""
    return os.getenv('K_SERVICE') is not None


def get_gcs_bucket() -> Optional[str]:
    """Obtém o bucket do GCS das variáveis de ambiente."""
    return os.getenv('GCS_BUCKET')


def download_model_from_gcs(bucket_name: str, model_path: Path) -> bool:
    """
    Baixa modelo treinado do Google Cloud Storage.
    
    Args:
        bucket_name: Nome do bucket GCS (ex: 'saude-chatbot-models')
        model_path: Caminho local para armazenar o modelo
        
    Returns:
        True se download foi bem-sucedido, False caso contrário
    """
    try:
        from google.cloud import storage
    except ImportError:
        logger.error("❌ google-cloud-storage não instalado!")
        logger.error("Execute: pip install google-cloud-storage")
        return False
    
    try:
        logger.info(f"📥 Baixando modelo de ML do bucket GCS: {bucket_name}")
        
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # Caminho remoto: models/melhor_modelo.keras
        remote_path = "models/melhor_modelo.keras"
        blob = bucket.blob(remote_path)
        
        if not blob.exists():
            logger.warning(f"⚠️  Modelo não encontrado em gs://{bucket_name}/{remote_path}")
            logger.warning("   Execute: gsutil cp Departamento_Medico/melhor_modelo.keras gs://seu-bucket/models/")
            return False
        
        # Criar diretório se não existir
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Baixar arquivo
        blob.download_to_filename(str(model_path))
        
        # Validar tamanho (modelo real tem ~193MB)
        file_size = model_path.stat().st_size
        if file_size < 1_000_000:  # Menos de 1MB é suspeito (pode ser ponteiro)
            logger.warning(f"⚠️  Arquivo pequeno demais: {file_size} bytes (esperado ~193MB)")
            return False
        
        logger.info(f"✅ Modelo baixado com sucesso! ({file_size / 1_000_000:.1f}MB)")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro ao baixar modelo do GCS: {str(e)}")
        return False


def upload_model_to_gcs(bucket_name: str, model_path: Path) -> bool:
    """
    Envia modelo treinado para Google Cloud Storage.
    Executado manualmente antes de primeiro deploy.
    
    Args:
        bucket_name: Nome do bucket GCS (ex: 'saude-chatbot-models')
        model_path: Caminho local do modelo a enviar
        
    Returns:
        True se upload foi bem-sucedido, False caso contrário
    """
    try:
        from google.cloud import storage
    except ImportError:
        logger.error("❌ google-cloud-storage não instalado!")
        logger.error("Execute: pip install google-cloud-storage")
        return False
    
    if not model_path.exists():
        logger.error(f"❌ Modelo não existe: {model_path}")
        return False
    
    # Validar que é arquivo real (não ponteiro Git LFS)
    file_size = model_path.stat().st_size
    if file_size < 1_000_000:  # Menos de 1MB é suspeito
        logger.error(f"❌ Arquivo muito pequeno: {file_size} bytes")
        logger.error("   Certifique-se que clonou com: git lfs pull")
        return False
    
    try:
        logger.info(f"📤 Enviando modelo para bucket GCS: {bucket_name}")
        
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # Caminho remoto
        remote_path = "models/melhor_modelo.keras"
        blob = bucket.blob(remote_path)
        
        blob.upload_from_filename(str(model_path))
        
        logger.info(f"✅ Modelo enviado com sucesso! ({file_size / 1_000_000:.1f}MB)")
        logger.info(f"   gs://{bucket_name}/{remote_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro ao enviar modelo para GCS: {str(e)}")
        return False


def sync_models_startup() -> bool:
    """
    Sincroniza modelo de ML no startup do container Cloud Run.
    NÃO faz nada localmente (assume Git LFS já configurado).
    
    Chamado automaticamente por init_container.py
    
    Returns:
        True se sincronização bem-sucedida, False caso contrário
    """
    # Verificar se é Cloud Run
    if not is_cloud_run():
        logger.info("ℹ️  Executando localmente - usando Git LFS para modelos")
        logger.info("    (certifique-se: git lfs install && git lfs pull)")
        return True
    
    # Em Cloud Run: baixar do GCS
    bucket_name = get_gcs_bucket()
    if not bucket_name:
        logger.warning("⚠️  GCS_BUCKET não configurada!")
        logger.warning("   Classificação de X-ray pode falhar")
        return False
    
    # Caminho onde o modelo será armazenado em Cloud Run
    model_path = Path("/app/Departamento_Medico/melhor_modelo.keras")
    
    logger.info("=" * 70)
    logger.info("🔄 Sincronizando Modelos com Google Cloud Storage...")
    logger.info("=" * 70)
    
    # Tentar baixar do GCS
    success = download_model_from_gcs(bucket_name, model_path)
    
    if success:
        logger.info("✅ Modelo pronto para usar!")
    else:
        logger.warning("⚠️  Modelo não disponível em GCS")
        logger.warning("   Classificação de raio-X não funcionará")
        logger.warning("")
        logger.warning("   SOLUÇÃO: Upload manual antes de deploy")
        logger.warning("   1. git lfs pull  (localmente)")
        logger.warning("   2. gsutil mb gs://seu-bucket-models")
        logger.warning("   3. GCS_BUCKET=seu-bucket-models python sync_models.py upload")
    
    return True


if __name__ == "__main__":
    """
    Script para upload manual do modelo para GCS.
    
    Uso:
        # Primeiro: instalar git-lfs e baixar arquivo real
        git lfs install
        git lfs pull
        
        # Depois: upload
        GCS_BUCKET=saude-chatbot-models python sync_models.py upload
    """
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if len(sys.argv) < 2:
        print("=" * 70)
        print("Script de Sincronização de Modelos com Google Cloud Storage")
        print("=" * 70)
        print("\nUSO: python sync_models.py [upload|check]")
        print("\nPRÉ-REQUISITOS PARA UPLOAD:")
        print("  1. Ter executado: git lfs install")
        print("  2. Ter baixado: git lfs pull")
        print("  3. Verificar arquivo real: file Departamento_Medico/melhor_modelo.keras")
        print("     (não deve ser 'ASCII text', deve ser 'Keras model')")
        print("  4. Ter credenciais GCS: gcloud auth application-default login")
        print("\nCOMPLETO:")
        print("  # Upload (primeira vez)")
        print("  git lfs install && git lfs pull")
        print("  gsutil mb gs://saude-chatbot-models  # criar bucket")
        print("  GCS_BUCKET=saude-chatbot-models python sync_models.py upload")
        print("\n  # Verificar")
        print("  GCS_BUCKET=saude-chatbot-models python sync_models.py check")
        print("\nDEPLOY:")
        print("  gcloud run deploy saude-chatbot --source . \\")
        print("    --set-env-vars GCS_BUCKET=saude-chatbot-models")
        sys.exit(1)
    
    action = sys.argv[1].lower()
    bucket = get_gcs_bucket()
    
    if not bucket:
        print("❌ Variável GCS_BUCKET não configurada!")
        print("Execute: export GCS_BUCKET=seu-bucket-gcs")
        sys.exit(1)
    
    local_model = Path("Departamento_Medico/melhor_modelo.keras")
    
    if action == "upload":
        if upload_model_to_gcs(bucket, local_model):
            print("\n" + "=" * 70)
            print("✅ Upload concluído com sucesso!")
            print("=" * 70)
            print("\nPróximo passo:")
            print(f"  gcloud run deploy saude-chatbot --source . \\")
            print(f"    --set-env-vars GCS_BUCKET={bucket}")
            print("\nO modelo será baixado automaticamente no startup do Cloud Run!")
            sys.exit(0)
        else:
            print("\n❌ Upload falhou! Verifique os erros acima.")
            sys.exit(1)
    
    elif action == "check":
        print("Verificando modelo no GCS...")
        if download_model_from_gcs(bucket, Path("/tmp/test_model.keras")):
            print("\n" + "=" * 70)
            print("✅ Modelo disponível e válido no GCS!")
            print("=" * 70)
            sys.exit(0)
        else:
            print("\n❌ Modelo não encontrado ou inválido no GCS!")
            sys.exit(1)
    
    else:
        print(f"❌ Ação desconhecida: {action}")
        print("Use: upload ou check")
        sys.exit(1)
