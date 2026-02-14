"""
Integra gravador.py e transcrever.py para uso no chatbot.
Grava audio com sounddevice e transcreve com Whisper API.
Processa videos de raio-X para classificacao de frames.
"""
import sys
import os
from pathlib import Path
import cv2
from PIL import Image
import shutil
from collections import Counter
import numpy as np

# Adiciona pastas ao path para imports
sys.path.insert(0, str(Path(__file__).parent / "Gravador"))
sys.path.insert(0, str(Path(__file__).parent / "Transcrever"))
sys.path.insert(0, str(Path(__file__).parent / "Video"))

from xray_classifier import get_classifier

#################################### VIDEO RAIO-X ####################################

MIN_CONFIDENCE_THRESHOLD = 0.4
TARGET_CLASSIFICATIONS_PER_SECOND = 2
MIN_CLASSIFY_INTERVAL = 3
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'webm'}


def allowed_video_file(filename):
    """Verifica se o arquivo tem extensao de video permitida."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS


def extract_xray_region(frame):
    """
    Detecta e extrai a regiao do raio-X de um frame de gravacao de tela.
    
    IGUAL AO ORIGINAL - sem mudanças
    """
    h, w = frame.shape[:2]
    frame_area = h * w

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return frame

    min_area = frame_area * 0.15
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    if not valid_contours:
        return frame

    largest = max(valid_contours, key=cv2.contourArea)
    x, y, rw, rh = cv2.boundingRect(largest)

    if (rw * rh) > frame_area * 0.9:
        return frame

    margin = 5
    x = max(0, x - margin)
    y = max(0, y - margin)
    rw = min(w - x, rw + 2 * margin)
    rh = min(h - y, rh + 2 * margin)

    return frame[y:y+rh, x:x+rw]


def enhance_xray_frame(frame):
    """
    Normaliza um frame de raio-X extraido de video.
    
    IGUAL AO ORIGINAL - sem mudanças
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def processar_video_xray(video_path, show_window=False):
    """
    Processa um video de raio-X, classificando frames periodicamente.

    *** ADAPTADO PARA CLOUD ***
    
    Args:
        video_path (str): Caminho absoluto para o arquivo de video.
        show_window (bool): Se True, exibe janela OpenCV (apenas local).
                           Se False, processa em background (cloud mode).

    Returns:
        dict com final_classification, frame_results, classification_counts, etc.
    """
    classifier = get_classifier()

    if not classifier.is_model_loaded():
        return {
            'success': False,
            'error': 'Modelo de classificacao nao carregado'
        }

    # Verificar se DISPLAY está disponível
    has_display = os.environ.get('DISPLAY') is not None
    can_show_window = show_window and has_display
    
    if show_window and not has_display:
        print("⚠️  AVISO: show_window=True mas DISPLAY não disponível (modo cloud)")
        print("   Processando sem exibir janela...")

    # Abrir vídeo
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return {
            'success': False,
            'error': f'Nao foi possivel abrir o video: {video_path}'
        }

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Amostragem adaptativa baseada no FPS real do video
        if fps > 0:
            classify_every_n = max(MIN_CLASSIFY_INTERVAL, int(fps / TARGET_CLASSIFICATIONS_PER_SECOND))
        else:
            classify_every_n = 10

        print(f"Video: {fps:.0f} FPS, {total_frames} frames, "
              f"classificando a cada {classify_every_n} frames "
              f"(~{fps/max(classify_every_n, 1):.1f} classificacoes/s)")
        
        if can_show_window:
            print("🖥️  Exibindo janela de preview...")
        else:
            print("☁️  Processando em background (cloud mode)...")

        frame_count = 0
        frame_results = []
        classification_names = []
        last_result = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Classificar a cada N frames
            if frame_count % classify_every_n == 0:
                xray_frame = extract_xray_region(frame)
                xray_frame = enhance_xray_frame(xray_frame)
                rgb_frame = cv2.cvtColor(xray_frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)

                result = classifier.classify(pil_image)

                if result['success']:
                    frame_results.append({
                        'frame_number': frame_count,
                        'class_name': result['class_name'],
                        'confidence': result['confidence'],
                        'all_probabilities': result['all_probabilities']
                    })
                    classification_names.append(result['class_name'])
                    last_result = result
                    print(f"Frame {frame_count}: {result['class_name']} "
                          f"({result['confidence']*100:.1f}%)")

            # *** ADAPTAÇÃO PRINCIPAL: Exibir apenas se show_window=True E display disponível ***
            if can_show_window and last_result is not None:
                # Sobrepor resultado da classificacao no frame
                class_name = last_result['class_name']
                confidence = last_result['confidence']
                label = f"{class_name}: {confidence*100:.1f}%"

                # Fundo escuro para legibilidade
                (text_w, text_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                cv2.rectangle(frame, (10, 10), (20 + text_w, 40 + text_h),
                              (0, 0, 0), -1)

                # Classe principal em verde
                cv2.putText(frame, label, (15, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                # Todas as probabilidades em branco
                y_offset = 70
                for cls_name, prob in last_result['all_probabilities'].items():
                    prob_text = f"{cls_name}: {prob*100:.1f}%"
                    cv2.putText(frame, prob_text, (15, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    y_offset += 25

                # Contador de frames
                cv2.putText(frame, f"Frame: {frame_count}/{total_frames}",
                            (15, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (200, 200, 200), 1)

                # *** EXIBIR JANELA (apenas se display disponível) ***
                cv2.imshow("Classificador de Raio-X", frame)

                # Check para fechar janela (apenas se janela aberta)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("⏹️  Processamento interrompido pelo usuário")
                    break

            frame_count += 1

        cap.release()
        
        # *** FECHAR JANELAS APENAS SE FORAM ABERTAS ***
        if can_show_window:
            cv2.destroyAllWindows()

        if not frame_results:
            return {
                'success': False,
                'error': 'Nenhum frame foi classificado com sucesso'
            }

        # Filtrar frames com confianca baixa
        reliable_results = [
            fr for fr in frame_results
            if fr['confidence'] >= MIN_CONFIDENCE_THRESHOLD
        ]

        if not reliable_results:
            reliable_results = frame_results

        filtered_count = len(frame_results) - len(reliable_results)
        if filtered_count > 0:
            print(f"Filtrados {filtered_count} frames com confianca "
                  f"abaixo de {MIN_CONFIDENCE_THRESHOLD*100:.0f}%")

        # Votacao ponderada
        class_labels = ['Covid-19', 'Normal', 'Pneumonia Viral', 'Pneumonia Bacteriana']
        weighted_scores = {label: 0.0 for label in class_labels}

        for fr in reliable_results:
            for label in class_labels:
                weighted_scores[label] += fr['all_probabilities'].get(label, 0.0)

        dominant_class = max(weighted_scores, key=weighted_scores.get)

        total_frames_used = len(reliable_results)
        avg_probabilities = {
            label: weighted_scores[label] / total_frames_used
            for label in class_labels
        }
        avg_confidence = avg_probabilities[dominant_class]

        class_counts = Counter(fr['class_name'] for fr in reliable_results)

        print(f"Votacao ponderada: {dominant_class} "
              f"(confianca media: {avg_confidence*100:.1f}%)")

        return {
            'success': True,
            'final_classification': {
                'success': True,
                'class_name': dominant_class,
                'confidence': avg_confidence,
                'all_probabilities': avg_probabilities
            },
            'total_frames_analyzed': len(frame_results),
            'total_frames_reliable': len(reliable_results),
            'total_frames_video': total_frames,
            'fps': fps,
            'frame_results': frame_results,
            'classification_counts': dict(class_counts)
        }

    except Exception as e:
        print(f"❌ Erro ao processar vídeo: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': f'Erro ao processar vídeo: {str(e)}'
        }
    finally:
        # Garantir que recursos sejam liberados
        try:
            cap.release()
        except:
            pass
        
        if can_show_window:
            try:
                cv2.destroyAllWindows()
            except:
                pass

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python gravar_e_transcrever.py <caminho_video> [show_window]")
        print("Exemplo: python gravar_e_transcrever.py video.mp4 false")
        sys.exit(1)
    
    video_path = sys.argv[1]
    show_window = len(sys.argv) > 2 and sys.argv[2].lower() == 'true'
    
    print(f"Processando: {video_path}")
    print(f"Exibir janela: {show_window}")
    
    result = processar_video_xray(video_path, show_window=show_window)
    
    if result['success']:
        print("\n✅ SUCESSO!")
        print(f"Classificação: {result['final_classification']['class_name']}")
        print(f"Confiança: {result['final_classification']['confidence']*100:.1f}%")
    else:
        print(f"\n❌ ERRO: {result['error']}")