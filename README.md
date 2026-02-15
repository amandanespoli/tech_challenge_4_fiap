# Classificação de Doenças Pulmonares - Chatbot Multimodal em Cloud

Sistema de classificação automática de imagens de raio-X torácico utilizando Deep Learning com interface de chatbot multimodal.

**Tech Challenge 6IADT - Fase 4 | FIAP Pos-Tech**

---

## Sobre o Projeto

Chatbot inteligente que integra processamento de imagens médicas com IA generativa para auxiliar na identificação de condições pulmonares a partir de raios-X. O sistema combina:

- **Classificação por Deep Learning** (ResNet50) para análise de imagens
- **RAG (Retrieval-Augmented Generation)** com ChromaDB para informações de saúde
- **Interação multimodal**: texto, áudio, imagem e vídeo
- **Interface web responsiva** com Flask
- **Cloud-ready**: preparado para Docker e ambientes serverless

> **IMPORTANTE:** Este sistema é apenas informativo e NÃO substitui o diagnóstico de um profissional de saúde qualificado.

---

## Classes de Classificação

| Classe | Descrição | Código |
|--------|-----------|--------|
| **Covid-19** | Pacientes diagnosticados com COVID-19 | 0 |
| **Normal** | Pacientes saudáveis (sem doença pulmonar) | 1 |
| **Pneumonia Viral** | Pneumonia causada por vírus | 2 |
| **Pneumonia Bacteriana** | Pneumonia causada por bactérias | 3 |

---

## Funcionalidades

### 1. Classificação de Raio-X por Imagem

Upload de imagens de raio-X torácico para classificação automática.

- **Formatos aceitos:** PNG, JPG, JPEG, GIF, BMP, WEBP
- **Validação:** GPT-4o Vision verifica se a imagem é realmente um raio-X torácico
- **Modelo:** ResNet50 com Transfer Learning (entrada 256x256 pixels)
- **Saída:** Classe predita, confiança e probabilidades de todas as classes

### 2. Classificação de Raio-X por Vídeo

Upload de vídeos contendo imagens de raio-X para análise frame a frame.

- **Formatos aceitos:** MP4, AVI, MOV, MKV, WMV, WEBM
- **Amostragem:** 1 classificação a cada 30 frames
- **Agregação:** Classe dominante (moda) + confiança média

### 3. Chat por Texto

Conversar com o chatbot usando processamento de linguagem natural.

- Perguntas sobre saúde e doenças pulmonares
- Follow-up sobre análises de raio-X realizadas
- Respostas contextualizadas baseadas em informações do RAG

### 4. Chat de Áudio

Gravação de áudio via navegador para interação por voz.

- Transcrição automática via Whisper API
- Processamento como mensagem de texto
- Resposta em áudio via Text-to-Speech

---

## Requisitos

- Python 3.9+
- OpenAI API Key (GPT-4o, Whisper, TTS)
- 2GB RAM mínimo

---

## Instalação

### 1. Clone o repositório

```bash
git clone <seu-repositorio>
cd tech_challenge_4_fiap
```

### 2. Configure variáveis de ambiente

Crie um arquivo `.env` na raiz do projeto:

```env
OPENAI_API_KEY=sk-xxx...
ENVIRONMENT=development
SECRET_KEY=sua-chave-secreta-aqui
```

### 3. Instale dependências

```bash
pip install -r requirements.txt
```

### 4. Inicie o servidor

```bash
python chatbot.py
```

O chatbot estará disponível em `http://localhost:5000`

---

## Uso com Docker

### Build da imagem

```bash
docker build -t chatbot-raio-x .
```

### Executar container

```bash
docker run -p 8080:8080 \
  -e OPENAI_API_KEY=sk-xxx... \
  -e ENVIRONMENT=production \
  chatbot-raio-x
```

### Com Docker Compose

```bash
docker-compose up
```

---

## Estrutura do Projeto

```
.
├── chatbot.py                    # Aplicação principal (Flask)
├── xray_classifier.py            # Classificador de raio-X
├── gravar_e_transcrever.py      # Processamento de vídeo
├── config.py                     # Configurações centralizadas
├── create_db.py                  # Script para criar ChromaDB
├── requirements.txt              # Dependências Python
├── Dockerfile                    # Build para containers
├── docker-compose.yml            # Orquestração Docker
├── cloudbuild.yaml               # CI/CD (Google Cloud Build)
├── settings.json                 # Configurações do usuário
├── .env                          # Variáveis de ambiente (não versionado)
│
├── templates/
│   └── index.html                # Interface web
├── static/
│   ├── bot-avatar.png           # Avatar do bot
│   ├── user-avatar.png          # Avatar do usuário
│   └── pdf_results.json         # Resultados de busca RAG
│
├── Departamento_Medico/
│   ├── melhor_modelo.keras      # Modelo ResNet50 treinado
│   └── README.md                # Documentação do modelo
│
├── chromasaude/                 # Base de dados ChromaDB
│   └── chroma.sqlite3           # Índice de embeddings
│
└── data/                        # Dados adicionais
```

---

## Endpoints Principais da API

### Classificação de Imagem
```
POST /upload_xray
Content-Type: multipart/form-data
```

### Classificação de Vídeo
```
POST /upload_video
Content-Type: multipart/form-data
```

### Chat de Texto
```
POST /send_message
Content-Type: application/json
Body: {"message": "..."}
```

### Chat de Áudio
```
POST /upload_audio
Content-Type: multipart/form-data
```

### Health Check
```
GET /health
```

---

## Variáveis de Ambiente

| Variável | Descrição | Padrão |
|----------|-----------|--------|
| `OPENAI_API_KEY` | Chave da API OpenAI | ⚠️ Obrigatória |
| `ENVIRONMENT` | Ambiente (development/production) | development |
| `SECRET_KEY` | Chave secreta para Flask | dev-secret-key |
| `PORT` | Porta do servidor | 5000 |

---

## Tecnologias

- **Backend:** Flask (Python 3.9+)
- **Modelo de Classificação:** ResNet50 (TensorFlow/Keras)
- **LLM:** OpenAI GPT-4o-mini
- **Speech-to-Text:** OpenAI Whisper
- **Text-to-Speech:** OpenAI TTS-1
- **Banco Vetorial:** ChromaDB + OpenAI Embeddings
- **Processamento de Vídeo:** OpenCV
- **Frontend:** HTML, CSS, JavaScript

---

## Licença

Desenvolvido para o Tech Challenge 6IADT - FIAP Pos-Tech
