from flask import Flask, render_template, request, jsonify
from PIL import Image
import threading
from openai import OpenAI
import wave
import io
import json
import os
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
import time
import atexit
import uuid
import logging

# LangChain (stack moderno)
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma

# Importar configurações
from config import (
    FEATURES, 
    FLASK_HOST, 
    FLASK_PORT, 
    FLASK_DEBUG,
    SECRET_KEY,
    MAX_CONTENT_LENGTH,
    UPLOAD_FOLDER,
    CHROMA_PATH,
    ALLOWED_IMAGE_EXTENSIONS,
    get_feature_status,
    is_feature_enabled
)

# Importar classificador de raio-X
from xray_classifier import get_classifier

# Importar processamento de vídeo
from gravar_e_transcrever import (
    processar_video_xray,
    allowed_video_file,
    ALLOWED_VIDEO_EXTENSIONS
)


# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configurações iniciais
app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

load_dotenv()

# Inicialização do cliente OpenAI
client = OpenAI()

# Variáveis globais
pergunta_num = 0
session_question_count = 0

class ChatBot:
    def __init__(self):
        self.frames = []
        self.frames_lock = threading.Lock()
        self.chat_history = []
        self.settings = self.load_settings()

        self.last_xray_result = None
        self.last_xray_timestamp = None

        logger.info("ChatBot inicializado")
        logger.info(f"Features disponíveis: {FEATURES}")

    def _initialize_audio_device(self):
        """
        Placeholder - áudio não disponível em container
        """
        logger.info("Audio device initialization skipped (cloud/docker mode)")
        pass

    def load_settings(self):
        settings_file = 'settings.json'
        if os.path.exists(settings_file):
            with open(settings_file, 'r') as file:
                return json.load(file)
        return {
            "selected_voice": "alloy",
            "hear_response": True
        }

    def transcribe_audio(self, audio_file):
        try:
            audio_file.seek(0)

            # Validate file size
            audio_file.seek(0, 2)
            file_size = audio_file.tell()
            audio_file.seek(0)

            if file_size < 1000:  # Less than 1KB
                logger.error(f"Audio file too small: {file_size} bytes")
                return ""

            logger.info(f"Transcribing audio: {file_size} bytes")

            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )

            logger.info(f"Transcription: {response.text}")
            return response.text
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""

    def get_response(self, user_message):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": """You are a helpful assistant that classifies user messages.
                        Your answer must be a JSON with two fields: "type" and "content".
                        The "content" field must ALWAYS be a string containing the user's question or message.

                        Classification rules:
                        - If it's a general question not related to health, type is 'normal' and content is your answer.
                        - If the user asks about health topics (saúde, doenças, sintomas, tratamentos, epidemiologia, etc.), type is 'saude' and content is the user's question as a string.
                        - If the user asks to click or point to something, type is 'click' and content must be in english starting with 'point to the...'.
                        - If the user asks about the screen or image, type is 'image' and content is the user's question as a string.
                        - If the user asks about a previous X-ray result, diagnosis, classification result, or wants more information about a detected condition (covid, pneumonia, etc.), type is 'xray_followup' and content is the user's question as a string.
                        - Keywords for xray_followup: "diagnóstico", "raio-x", "resultado", "classificação", "explique mais", "o que significa", "covid", "pneumonia", "pulmão", "radiografia", "condição detectada".

                        Example response: {"type": "saude", "content": "Qual a definição e epidemiologia?"}
                        Example response: {"type": "xray_followup", "content": "Explique mais sobre a pneumonia detectada"}"""},
                    {"role": "assistant", "content": "\n".join(self.chat_history)},
                    {"role": "user", "content": user_message}
                ]
            )

            json_response = json.loads(response.choices[0].message.content)

            if json_response.get('type') == 'normal':
                return {'type': 'normal', 'content': json_response.get('content')}

            elif json_response.get('type') == 'saude':
                rag_response = self.get_ragsaude_response(json_response.get('content'))
                # Extrair apenas o conteúdo da resposta do RAG - Saúde
                if isinstance(rag_response, dict):
                    return {'type': 'saude', 'content': rag_response.get('content', '')}
                return {'type': 'saude', 'content': str(rag_response)}

            elif json_response.get('type') == 'xray_followup':
                followup = self.get_xray_followup_response(json_response.get('content'))
                return {'type': 'xray_followup', 'content': followup}

        except Exception as e:
            return {'type': 'error', 'content': f"Sorry, I couldn't get a response. Error: {e}"}

#################################### SAÚDE ####################################
    def get_ragsaude_response(self, question):
        # Garantir que question é uma string
        if isinstance(question, dict):
            question = question.get('content', str(question))
        if not isinstance(question, str):
            question = str(question)

        # Caminho para a base de dados do Chroma
        CHROMA_PATH_SAUDE = "chromasaude"

        api_key = os.getenv('OPENAI_API_KEY')
        # Inicialização do Chroma e do modelo de embedding
        embedding_function = OpenAIEmbeddings(openai_api_key=api_key)
        db = Chroma(persist_directory=CHROMA_PATH_SAUDE, embedding_function=embedding_function)

        # Pesquisar no banco de dados
        results = db.similarity_search_with_relevance_scores(question, k=5)
        print("Resultados de relevância:", results)
        print_formatted_results(results)

        # Verifique se há resultados relevantes
        if len(results) == 0 or results[0][1] < 0.3:
            print("Nenhum resultado relevante encontrado ou abaixo do limiar de 0.3.")
            return {'type': 'saude', 'content': "Desculpe, não encontrei informações relevantes nos documentos carregados."}

        # Estrutura do prompt com o contexto preenchido
        PROMPT_TEMPLATE_SAUDE = """
        # Role
        Você é um assistente virtual de saúde, especializado em fornecer informações educativas sobre saúde, bem-estar e qualidade de vida.

        # Task
        Sua tarefa é interpretar a pergunta do usuário e fornecer uma resposta clara, precisa, educada e direta, mantendo um tom profissional e acolhedor.

        # Specifics
        - A resposta deve conter no máximo 1200 caracteres. Não ultrapasse esse limite.
        - Utilize apenas as informações contidas no contexto fornecido sobre saúde.
        - Se não houver informação suficiente no contexto para responder à pergunta, diga: "Desculpe, mas não consigo ajudar com as informações disponíveis. Por favor, consulte um profissional de saúde."
        - IMPORTANTE: Sempre inclua uma recomendação para que o usuário consulte um profissional de saúde (médico, enfermeiro, nutricionista, etc.) para diagnósticos, tratamentos ou orientações personalizadas.
        - Nunca forneça diagnósticos médicos ou prescrições de medicamentos.
        - Seja empático e compreensivo com as preocupações de saúde do usuário.

        # Context
        Use o seguinte contexto para responder à questão da forma mais clara e precisa possível.
        Contexto: {context}

        Pergunta: {question}
        Resposta:

        """
        # Concatenar o conteúdo dos documentos relevantes
        conteudo = "\n\n".join([doc.page_content for doc, _score in results])
        print("Conteúdo extraído para o contexto:", conteudo)

        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_SAUDE)
        prompt = prompt_template.format(context=conteudo, question=question)

        completion = client.chat.completions.create(
            temperature=0.5,
            model="gpt-4o-mini",
            max_tokens=1000,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": question},
            ],
        )

        return {'type': 'saude', 'content': completion.choices[0].message.content}

#################################### RAIO-X FOLLOW-UP ####################################
    def get_xray_followup_response(self, question):
        """
        Responde perguntas de follow-up sobre o último raio-X analisado.
        Combina o contexto da classificação com busca no ChromaDB.
        """
        # Verificar se há contexto de raio-X recente
        if self.last_xray_result is None:
            return "Não encontrei nenhuma análise de raio-X recente. Por favor, faça o upload de um raio-X ou pergunte sobre a sua tela com um raio-X aberto."

        # Verificar se o contexto não está muito antigo (30 minutos)
        if time.time() - self.last_xray_result.get('timestamp', 0) > 1800:
            return "A análise de raio-X anterior já expirou (mais de 30 minutos). Por favor, faça uma nova análise."

        # Obter classificação anterior
        classification = self.last_xray_result.get('classification', {})
        class_name = classification.get('class_name', 'desconhecida')
        confidence = classification.get('confidence', 0)

        # Criar contexto enriquecido para a busca
        enriched_query = f"{class_name} {question}"

        # Buscar informações adicionais no ChromaDB
        rag_response = self.get_ragsaude_response(enriched_query)
        additional_info = rag_response.get('content', '') if isinstance(rag_response, dict) else str(rag_response)

        # Construir resposta contextualizada
        FOLLOWUP_PROMPT = f"""Com base na análise de raio-X anterior que detectou {class_name} com {confidence*100:.1f}% de confiança,
responda a seguinte pergunta do usuário de forma clara e educativa.

Contexto da análise anterior:
- Classificação: {class_name}
- Confiança: {confidence*100:.1f}%
- Informações de saúde: {self.last_xray_result.get('health_info', 'Não disponível')}

Informações adicionais do banco de dados:
{additional_info}

Pergunta do usuário: {question}

Responda de forma clara, empática e sempre inclua a recomendação de consultar um profissional de saúde.
IMPORTANTE: Este é apenas um resultado informativo e NÃO substitui o diagnóstico de um médico."""

        completion = client.chat.completions.create(
            temperature=0.5,
            model="gpt-4o-mini",
            max_tokens=1000,
            messages=[
                {"role": "system", "content": FOLLOWUP_PROMPT},
                {"role": "user", "content": question},
            ],
        )

        return completion.choices[0].message.content

    def cleanup(self):
        """Cleanup resources - ADAPTADO para cloud"""
        logger.info("Cleanup: Nenhum recurso de áudio para liberar (cloud mode)")

def allowed_image_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

# Função para formatar resultados de busca
def print_formatted_results(results):
    formatted_results = []
    for i, (doc, score) in enumerate(results, 1):
        result = {
            "number": i,
            "length": len(doc.page_content),
            "score": score,
            "content": doc.page_content
        }
        formatted_results.append(result)

    # Save results to a JSON file
    with open('static/pdf_results.json', 'w', encoding='utf-8') as f:
        json.dump(formatted_results, f, ensure_ascii=False, indent=2)

# Rota para análise de raio-X
@app.route('/upload_xray', methods=['POST'])
def upload_xray():
    """
    Rota para upload e análise de imagem de raio-X.
    Retorna classificação da doença e informações de saúde relacionadas.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo enviado'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'Nenhum arquivo selecionado'}), 400

    if not allowed_image_file(file.filename):
        return jsonify({
            'error': 'Formato inválido',
            'message': 'Por favor, envie uma imagem (PNG, JPG, JPEG, GIF, BMP ou WEBP)'
        }), 400

    try:
        # Carregar imagem com PIL
        image = Image.open(file.stream)

        # Obter classificador (singleton)
        classifier = get_classifier()

        if not classifier.is_model_loaded():
            return jsonify({
                'error': 'Modelo não disponível',
                'message': 'O modelo de classificação de raio-X não foi carregado.'
            }), 500

        # Verificar se é uma imagem de raio-X usando GPT-4o Vision
        is_xray = classifier.is_xray_image(image)

        if not is_xray:
            return jsonify({
                'type': 'not_xray',
                'content': 'A imagem enviada não parece ser um raio-X de tórax. Por favor, envie uma radiografia de tórax válida.'
            })

        # Classificar o raio-X
        result = classifier.classify(image)

        if not result['success']:
            return jsonify({
                'error': 'Falha na classificação',
                'message': result.get('error', 'Erro desconhecido')
            }), 500

        # Buscar informações de saúde no ChromaDB
        disease_query = classifier.get_disease_query(result['class_name'])
        health_info = chatbot.get_ragsaude_response(disease_query)
        health_content = health_info.get('content', '') if isinstance(health_info, dict) else str(health_info)

        # Armazenar contexto para follow-up
        chatbot.last_xray_result = {
            'classification': result,
            'health_info': health_content,
            'timestamp': time.time()
        }
        chatbot.last_xray_timestamp = time.time()

        # Adicionar ao histórico do chat
        chatbot.chat_history.append(f"[Raio-X Analisado]: {result['class_name']} ({result['confidence']*100:.1f}% confiança)")

        logging.info(f"Raio-X classificado: {result['class_name']} ({result['confidence']*100:.1f}%)")

        return jsonify({
            'type': 'xray',
            'classification': result,
            'health_info': health_content,
            'follow_up_hint': 'Você pode perguntar mais sobre este diagnóstico.'
        })

    except Exception as e:
        logging.error(f"Erro ao processar raio-X: {e}")
        return jsonify({
            'error': 'Erro ao processar imagem',
            'message': str(e)
        }), 500

# Rota para análise de vídeo de raio-X
@app.route('/upload_video', methods=['POST'])
def upload_video():
    global session_question_count
    session_question_count += 1

    if 'video' not in request.files:
        return jsonify({'error': 'Nenhum vídeo fornecido'}), 400

    video_file = request.files['video']

    if video_file.filename == '':
        return jsonify({'error': 'Nome de arquivo vazio'}), 400

    if not allowed_video_file(video_file.filename):
        return jsonify({
            'error': f'Formato de vídeo não suportado. Use: {", ".join(ALLOWED_VIDEO_EXTENSIONS)}'
        }), 400

    try:
        # Salvar vídeo temporariamente
        filename = secure_filename(video_file.filename)
        video_path = UPLOAD_FOLDER / f"{uuid.uuid4()}_{filename}"
        video_file.save(str(video_path))

        logger.info(f"Processando vídeo: {video_path}")

        # ADAPTADO: Processar SEM exibir janela (show_window=False)
        result = processar_video_xray(str(video_path), show_window=False)

        # Limpar arquivo temporário
        if video_path.exists():
            video_path.unlink()

        if not result.get('success'):
            return jsonify({
                'error': result.get('error', 'Erro desconhecido ao processar vídeo')
            }), 500

        # Extrair classificacao final (agregada)
        final_classification = result['final_classification']

        # Buscar informacoes de saude no ChromaDB (mesmo padrao do /upload_xray)
        disease_query = get_classifier().get_disease_query(final_classification['class_name'])
        health_info = chatbot.get_ragsaude_response(disease_query)
        health_content = health_info.get('content', '') if isinstance(health_info, dict) else str(health_info)

        # Armazenar contexto para follow-up (mesmo padrao do /upload_xray)
        chatbot.last_xray_result = {
            'classification': final_classification,
            'health_info': health_content,
            'timestamp': time.time()
        }
        chatbot.last_xray_timestamp = time.time()

        # Adicionar ao historico do chat
        chatbot.chat_history.append(
            f"[Video Raio-X Analisado]: {final_classification['class_name']} "
            f"({final_classification['confidence']*100:.1f}% confianca, "
            f"{result['total_frames_analyzed']} frames analisados)"
        )

        logging.info(
            f"Video raio-X classificado: {final_classification['class_name']} "
            f"({final_classification['confidence']*100:.1f}%) - "
            f"{result['total_frames_analyzed']} frames"
        )

        return jsonify({
            'type': 'video_xray',
            'classification': result['final_classification'],
            'stats': {
                'total_frames_analyzed': result['total_frames_analyzed'],
                'total_frames_reliable': result['total_frames_reliable'],
                'total_frames_video': result['total_frames_video'],
                'fps': result['fps'],
                'classification_counts': result['classification_counts']
            },
            'content': f"# Análise de Vídeo Concluída\\n\\n..."  # Igual ao original
        })

    except Exception as e:
        logger.error(f"Erro ao processar vídeo: {e}")
        return jsonify({
            'error': 'Erro ao processar vídeo',
            'message': str(e)
        }), 500

@app.route('/text_to_speech', methods=['POST'])
def text_to_speech():
    try:
        data = request.json
        text = data.get('text', '')
        
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text,
            response_format="mp3"  # Usar MP3 ao invés de WAV
        )
        
        # Retornar áudio como base64 para o navegador reproduzir
        audio_data = response.content
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        return jsonify({
            'status': 'success',
            'audio': audio_base64,
            'format': 'mp3'
        })
    except Exception as e:
        logger.error(f"Text-to-speech error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/save_settings', methods=['POST'])
def save_settings():
    data = request.json
    chatbot.settings.update(data)
    with open('settings.json', 'w') as file:
        json.dump(chatbot.settings, file)
    return jsonify({'status': 'success'})

# Instância global do chatbot
chatbot = ChatBot()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    data = request.json
    message = data.get('message', '')
    
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    chatbot.chat_history.append(f"You: {message}")
    response = chatbot.get_response(message)

    if response.get('type') == 'xray_screen':
        history_text = f"[Raio-X Detectado]: {response['classification']['class_name']}"
    else:
        history_text = response.get('content', str(response))
    chatbot.chat_history.append(f"Bot: {history_text}")

    return jsonify(response)

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    """
    NOVO: Recebe áudio gravado no navegador (WebRTC) e processa
    Substitui /start_recording e /stop_recording
    """
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    
    if audio_file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    try:
        # Salvar temporariamente
        temp_path = UPLOAD_FOLDER / f"{uuid.uuid4()}.wav"
        audio_file.save(str(temp_path))
        
        # Transcrever
        with open(temp_path, 'rb') as f:
            transcript = chatbot.transcribe_audio(f)
        
        if not transcript:
            return jsonify({
                'error': 'Transcription failed',
                'message': 'Could not understand the audio. Please try again.'
            }), 400
        
        # Processar mensagem
        chatbot.chat_history.append(f"You: {transcript}")
        response = chatbot.get_response(transcript)
        
        # Adicionar ao histórico
        if isinstance(response, dict):
            if response.get('type') == 'xray_screen':
                history_text = f"[Raio-X Detectado]: {response['classification']['class_name']}"
            else:
                history_text = response.get('content', str(response))
        else:
            history_text = str(response)
        
        chatbot.chat_history.append(f"Bot: {history_text}")
        
        # Limpar arquivo temporário
        if temp_path.exists():
            temp_path.unlink()
        
        return jsonify({
            'transcript': transcript,
            'response': response
        })
        
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        return jsonify({
            'error': 'Processing failed',
            'message': str(e)
        }), 500
    finally:
        # Garantir limpeza
        if temp_path.exists():
            try:
                temp_path.unlink()
            except:
                pass

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint para Docker/Kubernetes
    """
    try:
        # Verificar se modelo está carregado
        classifier = get_classifier()
        model_loaded = classifier.is_model_loaded()
        
        # Verificar se ChromaDB está acessível
        chroma_ok = CHROMA_PATH.exists()
        
        status = {
            'status': 'healthy' if model_loaded and chroma_ok else 'degraded',
            'model_loaded': model_loaded,
            'chroma_accessible': chroma_ok,
            'features': FEATURES,
            'environment': os.getenv('ENVIRONMENT', 'unknown')
        }
        
        status_code = 200 if status['status'] == 'healthy' else 503
        return jsonify(status), status_code
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 503

@app.route('/api/features', methods=['GET'])
def api_features():
    """
    Retorna quais features estão disponíveis
    Frontend pode esconder botões de features não disponíveis
    """
    return jsonify(get_feature_status())

def cleanup_on_exit():
    """Cleanup function called on program exit"""
    logger.info("Shutting down...")
    chatbot.cleanup()

# Instância global do chatbot (ANTES do atexit.register)
chatbot = ChatBot()

atexit.register(cleanup_on_exit)

if __name__ == '__main__':
    # Validar configuração no startup
    from config import validate_config
    validate_config()
    
    # Inicializar classificador (carrega modelo)
    logger.info("Inicializando classificador...")
    classifier = get_classifier()
    if classifier.is_model_loaded():
        logger.info("✅ Modelo carregado com sucesso!")
    else:
        logger.error("❌ AVISO: Modelo não foi carregado!")
    
    # Rodar aplicação
    logger.info(f"Iniciando servidor em {FLASK_HOST}:{FLASK_PORT}")
    logger.info(f"Ambiente: {os.getenv('ENVIRONMENT', 'development')}")
    logger.info(f"Features: {sum(FEATURES.values())}/{len(FEATURES)} disponíveis")
    
    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=FLASK_DEBUG
    )