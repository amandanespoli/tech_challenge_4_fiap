#!/usr/bin/env python3
"""
Módulo de sincronização ChromaDB com Google Cloud Storage.
Permite baixar/enviar o banco de dados vectorizado para persistência em Cloud Run.
"""
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

def is_cloud_run() -> bool:
    """Detecta se está rodando em Cloud Run (Google Cloud)."""
    return os.getenv('K_SERVICE') is not None or os.getenv('IS_CLOUD_RUN') == 'true'


def get_gcs_bucket() -> Optional[str]:
    """Obtém o bucket do GCS das variáveis de ambiente."""
    return os.getenv('GCS_BUCKET')


def download_chromadb_from_gcs(bucket_name: str, chroma_path: Path) -> bool:
    """
    Baixa o banco de dados ChromaDB do Google Cloud Storage.
    
    Args:
        bucket_name: Nome do bucket GCS (ex: 'saude-chatbot-db')
        chroma_path: Caminho local para armazenar os arquivos
        
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
        logger.info(f"📥 Baixando ChromaDB do bucket GCS: {bucket_name}")
        
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # Listar blobs no bucket com prefixo correto
        blobs = list(bucket.list_blobs(prefix='chromasaude/'))
        
        if not blobs:
            logger.warning(f"⚠️  Nenhum arquivo encontrado em gs://{bucket_name}/chromasaude/")
            logger.warning("   Execute: GCS_BUCKET={} python sync_chromadb.py upload".format(bucket_name))
            return False
        
        # Criar diretório se não existir
        chroma_path.mkdir(parents=True, exist_ok=True)
        
        # Baixar cada arquivo
        files_downloaded = 0
        for blob in blobs:
            # Obter caminho relativo (remover 'chromasaude/' do prefixo)
            relative_path = blob.name[len('chromasaude/'):]
            
            if relative_path:  # Ignorar a pasta vazia
                local_file_path = chroma_path / relative_path
                local_file_path.parent.mkdir(parents=True, exist_ok=True)
                
                blob.download_to_filename(str(local_file_path))
                logger.debug(f"  ✅ {relative_path}")
                files_downloaded += 1
        
        logger.info(f"✅ ChromaDB baixado com sucesso! ({files_downloaded} arquivo(s))")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro ao baixar ChromaDB do GCS: {str(e)}")
        return False


def upload_chromadb_to_gcs(bucket_name: str, chroma_path: Path) -> bool:
    """
    Envia o banco de dados ChromaDB para Google Cloud Storage.
    
    Args:
        bucket_name: Nome do bucket GCS (ex: 'saude-chatbot-db')
        chroma_path: Caminho local do ChromaDB a enviar
        
    Returns:
        True se upload foi bem-sucedido, False caso contrário
    """
    try:
        from google.cloud import storage
    except ImportError:
        logger.error("❌ google-cloud-storage não instalado!")
        logger.error("Execute: pip install google-cloud-storage")
        return False
    
    if not chroma_path.exists():
        logger.error(f"❌ Diretório não existe: {chroma_path}")
        return False
    
    # Validar que tem arquivos
    files = list(chroma_path.rglob('*'))
    if not files or all(f.is_dir() for f in files):
        logger.error(f"❌ Diretório vazio: {chroma_path}")
        logger.error("   Execute 'python create_db.py' primeiro com PDFs em 'data/'")
        return False
    
    try:
        logger.info(f"📤 Enviando ChromaDB para bucket GCS: {bucket_name}")
        
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # Fazer upload recursivo de todos os arquivos
        files_uploaded = 0
        for local_file_path in chroma_path.rglob('*'):
            if local_file_path.is_file():
                # Gerar caminho remoto (chromasaude/arquivo)
                relative_path = local_file_path.relative_to(chroma_path)
                blob_path = f"chromasaude/{relative_path}"
                
                blob = bucket.blob(blob_path)
                blob.upload_from_filename(str(local_file_path))
                logger.debug(f"  ✅ {relative_path}")
                files_uploaded += 1
        
        logger.info(f"✅ ChromaDB enviado com sucesso! ({files_uploaded} arquivo(s))")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro ao enviar ChromaDB para GCS: {str(e)}")
        return False


def sync_chromadb_startup() -> bool:
    """
    Sincroniza ChromaDB no startup do container Cloud Run.
    Chamado automaticamente por init_container.py se IS_CLOUD_RUN=true
    
    Returns:
        True se sincronização bem-sucedida, False caso contrário
    """
    if not is_cloud_run():
        logger.info("ℹ️ Não está em Cloud Run, pulando sincronização GCS")
        return True
    
    bucket_name = get_gcs_bucket()
    if not bucket_name:
        logger.error("❌ Variável GCS_BUCKET não configurada!")
        logger.error("Configure: gcloud run deploy ... --set-env-vars GCS_BUCKET=seu-bucket")
        return False
    
    # Usar caminho consistente com config.py
    chroma_path = Path("/app/chromasaude")
    
    logger.info("=" * 70)
    logger.info("🔄 Sincronizando ChromaDB com Google Cloud Storage...")
    logger.info("=" * 70)
    
    # Tentar baixar do GCS
    success = download_chromadb_from_gcs(bucket_name, chroma_path)
    
    if success:
        logger.info("✅ ChromaDB pronto para usar!")
    else:
        logger.warning("⚠️  ChromaDB não disponível em GCS")
        logger.warning("   Prosseguindo, mas RAG não terá contexto médico")
        logger.warning("")
        logger.warning("   PRÓXIMOS PASSOS:")
        logger.warning("   1. python create_db.py  (com PDFs em data/)")
        logger.warning("   2. GCS_BUCKET={} python sync_chromadb.py upload".format(bucket_name))
        logger.warning("   3. Redeploy no Cloud Run")
    
    return True


if __name__ == "__main__":
    """
    Script para upload manual do ChromaDB para GCS.
    
    Uso:
        # Upload para produção
        python sync_chromadb.py upload
        
        # Download do GCS
        python sync_chromadb.py download
    """
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if len(sys.argv) < 2:
        print("Uso: python sync_chromadb.py [upload|download]")
        print("\nPRÉ-REQUISITOS:")
        print("  1. Ter executado: python create_db.py")
        print("  2. chromasaude/ deve ter arquivos")
        print("  3. Ter credenciais GCS: gcloud auth application-default login")
        print("\nEXEMPLOS:")
        print("  # Upload do ChromaDB local para GCS")
        print("  GCS_BUCKET=saude-chatbot-db python sync_chromadb.py upload")
        print("\n  # Download do GCS para testar")
        print("  GCS_BUCKET=saude-chatbot-db python sync_chromadb.py download")
        sys.exit(1)
    
    action = sys.argv[1].lower()
    bucket = get_gcs_bucket()
    
    if not bucket:
        print("❌ Variável GCS_BUCKET não configurada!")
        print("Export: export GCS_BUCKET=seu-bucket-gcs")
        sys.exit(1)
    
    local_chroma = Path("chromasaude")
    
    if action == "upload":
        if upload_chromadb_to_gcs(bucket, local_chroma):
            print("\n" + "=" * 70)
            print("✅ Upload concluído com sucesso!")
            print("=" * 70)
            print("\nAgora você pode:")
            print("  1. Deploy em Cloud Run: gcloud run deploy saude-chatbot --source .")
            print("  2. O ChromaDB será baixado automaticamente no startup")
            sys.exit(0)
        else:
            print("\n❌ Upload falhou! Verifique os erros acima.")
            sys.exit(1)
    
    elif action == "download":
        if download_chromadb_from_gcs(bucket, local_chroma):
            print("\n" + "=" * 70)
            print("✅ Download concluído com sucesso!")
            print("=" * 70)
            sys.exit(0)
        else:
            print("\n❌ Download falhou! Verifique os erros acima.")
            sys.exit(1)
    
    else:
        print(f"❌ Ação desconhecida: {action}")
        print("Use: upload ou download")
        sys.exit(1)
