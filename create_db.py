import os
import sys
import traceback
import time
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

import openai

# Configurar logging estruturado para cloud
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

load_dotenv()

# PATHS CLOUD-READY - Suporta desenvolvimento local e container
BASE_DIR = Path(__file__).parent
IS_DOCKER = os.path.exists('/.dockerenv')

if IS_DOCKER:
    # Paths para container
    CHROMA_PATH = Path("/app/chromasaude")
    DATA_PATH = Path("/app/data")
    CHECKPOINT_FILE = Path("/tmp/chroma_checkpoint.json")  # /tmp é persistente apenas durante execução
else:
    # Paths para desenvolvimento local
    CHROMA_PATH = BASE_DIR / "chromasaude"
    DATA_PATH = BASE_DIR / "data"
    CHECKPOINT_FILE = BASE_DIR / ".chroma_checkpoint.json"

# Criar diretórios se não existirem
CHROMA_PATH.mkdir(parents=True, exist_ok=True)
DATA_PATH.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 100
PERSIST_FREQUENCY = 25
MAX_RETRIES = 3
RETRY_DELAY = 2  # segundos

logger.info(f"Environment: {'Docker' if IS_DOCKER else 'Local'}")
logger.info(f"Chroma path: {CHROMA_PATH}")
logger.info(f"Data path: {DATA_PATH}")

def main(skip_test=False):
    """
    Função principal para criar/atualizar ChromaDB.
    
    Args:
        skip_test: Se True, pula o teste do banco (útil em CI/CD)
    
    Returns:
        bool: True se sucesso, False se falha
    """
    try:
        logger.info("=" * 60)
        logger.info("Iniciando criação/atualização do ChromaDB")
        logger.info("=" * 60)
        
        # Validar API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("❌ OPENAI_API_KEY não configurada!")
        logger.info("✅ API Key encontrada")
        
        # Carregar documentos
        documents = load_documents()
        if not documents:
            logger.warning("⚠️ Nenhum documento PDF encontrado em {}".format(DATA_PATH))
            logger.warning("Para usar o RAG, adicione arquivos .pdf em data/")
            return False
        
        logger.info(f"✅ {len(documents)} documento(s) carregado(s)")
        
        # Dividir em chunks
        chunks = split_text(documents)
        
        # Processar chunks
        process_chunks(chunks)
        
        # Teste opcional
        if not skip_test:
            test_database()
        
        logger.info("=" * 60)
        logger.info("✅ ChromaDB criado/atualizado com sucesso!")
        logger.info("=" * 60)
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro durante execução: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def load_documents():
    """
    Carrega arquivos PDF recursivamente do diretório DATA_PATH.
    
    Returns:
        list: Lista de documentos carregados
    """
    logger.info(f"Carregando PDFs de {DATA_PATH}...")
    
    # Validar se pasta existe e tem arquivos
    if not DATA_PATH.exists():
        logger.warning(f"Pasta {DATA_PATH} não existe")
        return []
    
    pdf_files = list(DATA_PATH.glob("**/*.pdf"))
    if not pdf_files:
        logger.warning(f"Nenhum arquivo .pdf encontrado em {DATA_PATH}")
        return []
    
    logger.info(f"Encontrados {len(pdf_files)} arquivo(s) PDF")
    
    loader = DirectoryLoader(str(DATA_PATH), glob="**/*.pdf", loader_cls=PyPDFLoader)
    
    documents = []
    try:
        documents = loader.load()
        logger.info(f"✅ {len(documents)} página(s) carregada(s) dos PDFs")
    except Exception as e:
        logger.error(f"❌ Erro ao carregar documentos: {e}")
        raise
    
    return documents

def split_text(documents):
    """
    Divide documentos em chunks com sobreposição para melhor contexto.
    
    Args:
        documents: Lista de documentos do LangChain
        
    Returns:
        list: Lista de chunks
    """
    logger.info("Dividindo documentos em chunks...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
    )
    
    # Garantir que todos os documentos são instâncias Document
    if documents and isinstance(documents[0], dict):
        logger.warning("Convertendo documentos dict para Document objects...")
        documents = [Document(page_content=doc.get('content', '')) for doc in documents]
    
    chunks = text_splitter.split_documents(documents)
    logger.info(f"✅ {len(chunks)} chunck(s) criado(s)")
    logger.info(f"   - Exemplo: '{chunks[0].page_content[:100]}...'")
    
    return chunks

def process_chunks(chunks):
    """
    Processa chunks em lotes e armazena embeddings no ChromaDB.
    Implementa retry logic para falhas de API e checkpoint para retomada.
    
    Args:
        chunks: Lista de chunks de texto
    """
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("❌ OPENAI_API_KEY não configurada!")
    
    logger.info("Criando embedding function...")
    embedding_function = OpenAIEmbeddings(openai_api_key=api_key)
    
    # Carregar último checkpoint
    last_processed = load_checkpoint()
    
    if last_processed >= len(chunks):
        logger.warning(f"Checkpoint inválido ({last_processed} >= {len(chunks)}). Resetando...")
        last_processed = 0
        save_checkpoint(0)
    
    if last_processed > 0:
        logger.info(f"Retomando do checkpoint: chunk {last_processed}/{len(chunks)}")
    
    try:
        # Conectar ou criar ChromaDB
        if CHROMA_PATH.exists() and list(CHROMA_PATH.glob("*")):
            logger.info(f"✅ Banco existente encontrado em {CHROMA_PATH}")
            db = Chroma(persist_directory=str(CHROMA_PATH), embedding_function=embedding_function)
        else:
            logger.info(f"📦 Criando novo banco em {CHROMA_PATH}")
            db = Chroma(persist_directory=str(CHROMA_PATH), embedding_function=embedding_function)
            last_processed = 0
        
        # Processar chunks em lotes
        logger.info(f"Processando {len(chunks) - last_processed} chunk(s)...")
        
        for i in range(last_processed, len(chunks), BATCH_SIZE):
            batch = chunks[i:i+BATCH_SIZE]
            process_batch(db, batch, i, len(chunks))
        
        # Verificação final
        try:
            count = db._collection.count()
            logger.info(f"✅ Total de documentos no banco: {count}")
        except:
            logger.info("✅ Chunks processados com sucesso")
            
    except Exception as e:
        logger.error(f"❌ Erro ao processar chunks: {e}")
        logger.error(traceback.format_exc())
        raise

def process_batch(db, batch, start_index, total_chunks):
    """
    Processa um lote de chunks com retry logic.
    
    Args:
        db: Instância do ChromaDB
        batch: Lista de chunks
        start_index: Índice inicial do batch
        total_chunks: Total de chunks para progresso
    """
    for i, chunk in enumerate(batch):
        overall_index = start_index + i
        
        if overall_index % 10 == 0:
            progress = (overall_index / total_chunks) * 100
            logger.info(f"Progresso: {overall_index + 1}/{total_chunks} ({progress:.1f}%)")
        
        # Retry logic para falhas de API
        success = False
        for attempt in range(MAX_RETRIES):
            try:
                start_time = time.time()
                db.add_documents([chunk])
                elapsed = time.time() - start_time
                
                if elapsed > 2:
                    logger.debug(f"Chunk {overall_index + 1} demorou {elapsed:.2f}s")
                
                success = True
                break
                
            except openai.RateLimitError as e:
                logger.warning(f"Rate limit no chunk {overall_index + 1} (tentativa {attempt + 1}/{MAX_RETRIES})")
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                    logger.info(f"Aguardando {delay}s antes de retry...")
                    time.sleep(delay)
                else:
                    raise
                    
            except openai.APIError as e:
                logger.warning(f"Erro da API no chunk {overall_index + 1}: {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    raise
        
        if not success:
            logger.error(f"❌ Falha ao processar chunk {overall_index + 1} após {MAX_RETRIES} tentativas")
            raise RuntimeError(f"Falha irrecuperável no chunk {overall_index + 1}")
        
        # Salvar checkpoint periodicamente
        if (overall_index + 1) % PERSIST_FREQUENCY == 0:
            save_checkpoint(overall_index + 1)
            logger.debug(f"Checkpoint salvo: {overall_index + 1}/{total_chunks}")

def load_checkpoint():
    """Carrega último checkpoint de progresso."""
    try:
        if CHECKPOINT_FILE.exists():
            with open(CHECKPOINT_FILE, 'r') as f:
                data = json.load(f)
                return data.get('last_processed_chunk', 0)
    except Exception as e:
        logger.warning(f"Erro ao carregar checkpoint: {e}")
    
    return 0

def save_checkpoint(last_processed_chunk):
    """Salva progresso em checkpoint."""
    try:
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump({'last_processed_chunk': last_processed_chunk}, f)
    except Exception as e:
        logger.warning(f"Erro ao salvar checkpoint: {e}")

def test_database():
    """Testa a integridade do ChromaDB com queries de exemplo."""
    logger.info("\n" + "=" * 60)
    logger.info("Testando banco de dados")
    logger.info("=" * 60)
    
    try:
        api_key = os.getenv('OPENAI_API_KEY')
        embedding_function = OpenAIEmbeddings(openai_api_key=api_key)
        db = Chroma(persist_directory=str(CHROMA_PATH), embedding_function=embedding_function)
        
        # Teste 1: Quantidade de documentos
        count = db._collection.count()
        logger.info(f"✅ Documentos armazenados: {count}")
        
        if count == 0:
            logger.warning("⚠️ Banco vazio! Adicione PDFs em data/ e reexecute.")
            return
        
        # Teste 2: Similarity search
        test_queries = [
            "saúde pulmonar",
            "pneumonia",
            "doença respiratória"
        ]
        
        logger.info("\nTestando buscas semânticas:")
        for query in test_queries:
            try:
                results = db.similarity_search_with_relevance_scores(query, k=2)
                if results:
                    logger.info(f"  '{query}': {len(results)} resultado(s) (score: {results[0][1]:.2f})")
                else:
                    logger.warning(f"  '{query}': nenhum resultado")
            except Exception as e:
                logger.warning(f"  '{query}': erro na busca - {e}")
        
        # Teste 3: RAG com LLM
        logger.info("\nTestando RAG com modelo:")
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        query = "O que é pneumonia?"
        results = db.similarity_search_with_relevance_scores(query, k=3)
        
        if results:
            context = "\n".join([doc.page_content for doc, _ in results])
            prompt = ChatPromptTemplate.from_template(
                "Com base no contexto fornecido, responda brevemente:\n\n"
                "Contexto: {context}\n\n"
                "Pergunta: {question}\n\n"
                "Resposta (máximo 2 linhas):"
            )
            
            response = llm.invoke(prompt.format(context=context[:500], question=query))
            logger.info(f"  Resposta: {response.content[:200]}...")
        
        logger.info("\n✅ Testes concluídos com sucesso!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Erro durante testes: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    """
    Execute como:
        python create_db.py           # Criação normal com testes
        python create_db.py --no-test # Sem testes (CI/CD)
        python create_db.py --help    # Ajuda
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Criar/atualizar ChromaDB com embeddings de PDFs")
    parser.add_argument('--no-test', action='store_true', help='Pular testes do banco')
    args = parser.parse_args()
    
    success = main(skip_test=args.no_test)
    
    # Exit com código apropriado para CI/CD
    sys.exit(0 if success else 1)
