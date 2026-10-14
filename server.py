import base64
import json
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Servidor OMR - Centralize Docente")

# Dicionário de mapeamento de índices para letras
MAPA_LETRAS = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}

class PayloadOMR(BaseModel):
    idCurso: str
    disciplina: str
    numeroQuestoes: int
    pontuacaoPorQuestao: float
    gabaritoOficial: str  # Exemplo: '{"1":"A","2":"C","3":"B"}'
    imagemBase64: str

def alinhar_e_recortar_folha(img):
    """
    Localiza os 4 marcadores pretos nos cantos (marcas fiduciais)
    e aplica a Transformação de Perspectiva para desentortar a imagem.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Encontra contornos dos marcadores
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    quadrados = []
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        
        # Filtra por contornos de 4 vértices (quadrados/retângulos) com área mínima
        if len(approx) == 4 and cv2.contourArea(c) > 100:
            x, y, w, h = cv2.boundingRect(approx)
            ar = w / float(h)
            if 0.8 <= ar <= 1.2:  # Formato próximo de um quadrado
                quadrados.append((x, y, w, h, approx))

    # Ordena para pegar os 4 maiores marcadores dos cantos
    quadrados = sorted(quadrados, key=lambda b: b[2] * b[3], reverse=True)[:4]

    if len(quadrados) < 4:
        raise ValueError("Não foi possível localizar os 4 marcadores de canto na folha. Verifique o enquadramento e iluminação.")

    # Extrai os centros dos 4 marcadores
    pontos = []
    for q in quadrados:
        M = cv2.moments(q[4])
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            pontos.append([cx, cy])

    pts = np.array(pontos, dtype="float32")

    # Ordena os pontos: [superior-esquerdo, superior-direito, inferior-direito, inferior-esquerdo]
    s = pts.sum(axis=1)
    rect = np.zeros((4, 2), dtype="float32")
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # Define dimensões fixas padrão para a folha alinhada (A4 proporcional)
    largura, altura = 1000, 1414
    dst = np.array([
        [0, 0],
        [largura - 1, 0],
        [largura - 1, altura - 1],
        [0, altura - 1]
    ], dtype="float32")

    # Matriz de transformação de perspectiva
    M_persp = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(img, M_persp, (largura, altura))

    return warp

def avaliar_questoes_omr(img_alinhada, total_questoes, gabarito_dict):
    """
    Examina as regiões das bolhas na folha alinhada e conta a densidade de pixels pretos.
    """
    gray = cv2.cvtColor(img_alinhada, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    metade = (total_questoes + 1) // 2
    acertos = 0
    respostas_detectadas = {}

    # Regiões de interesse (ROI) aproximadas em percentuais da folha alinhada
    # Ajuste fino das coordenadas baseadas no layout do PDF gerado no Flutter
    largura, altura = img_alinhada.shape[1], img_alinhada.shape[0]

    for q in range(1, total_questoes + 1):
        # Define se a questão está na coluna esquerda ou direita
        if q <= metade:
            coluna = 0
            linha_index = q - 1
        else:
            coluna = 1
            linha_index = q - metade - 1

        # CÁLCULO DE COORDENADAS DINÂMICAS DAS BOLHAS
        # Coluna 0 (Esquerda): X inicia ~18% da largura | Coluna 1 (Direita): X inicia ~65%
        x_inicio_base = 0.18 if coluna == 0 else 0.65
        y_inicio_base = 0.28 + (linha_index * (0.60 / metade))  # Distribui verticalmente

        marcatura_maxima = 0
        opcao_marcada = None

        for idx_alt in range(5):  # 0: A, 1: B, 2: C, 3: D, 4: E
            # Deslocamento horizontal por alternativa
            x_bolha = int((x_inicio_base + (idx_alt * 0.055)) * largura)
            y_bolha = int(y_inicio_base * altura)
            r = int(0.012 * largura)  # Raio da região da bolha

            # Máscara circular para contar pixels marcados a caneta
            mascara = np.zeros(thresh.shape, dtype="uint8")
            cv2.circle(mascara, (x_bolha, y_bolha), r, 255, -1)
            
            # Conta pixels pretos dentro do círculo
            total_pixels_preenchidos = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mascara))

            if total_pixels_preenchidos > marcatura_maxima:
                marcatura_maxima = total_pixels_preenchidos
                opcao_marcada = MAPA_LETRAS[idx_alt]

        # Limiar mínimo de pixels para considerar como marcacao válida (evita ruídos)
        area_minima_preenchimento = (np.pi * (r ** 2)) * 0.35
        if marcatura_maxima >= area_minima_preenchimento:
            respostas_detectadas[str(q)] = opcao_marcada
            if gabarito_dict.get(str(q)) == opcao_marcada:
                acertos += 1
        else:
            respostas_detectadas[str(q)] = "N/A"  # Questão em branco ou marcação fraca

    return acertos, respostas_detectadas

@app.post("/corrigir_omr_ia")
async def corrigir_omr(payload: PayloadOMR):
    try:
        # 1. Decodifica a imagem Base64
        bytes_imagem = base64.b64decode(payload.imagemBase64)
        nparr = np.frombuffer(bytes_imagem, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Imagem inválida ou corrompida.")

        # 2. Converte a string do gabarito em dicionário Python
        gabarito = json.loads(payload.gabaritoOficial)

        # 3. Alinha a imagem usando os marcadores fiduciais nos cantos
        img_alinhada = alinhar_e_recortar_folha(img)

        # 4. Processa as bolhas de resposta
        acertos, respostas = avaliar_questoes_omr(
            img_alinhada, 
            payload.numeroQuestoes, 
            gabarito
        )

        nota_final = acertos * payload.pontuacaoPorQuestao

        return {
            "status": "SUCESSO",
            "acertos": acertos,
            "notaFinal": round(nota_final, 2),
            "respostasDetectadas": respostas,
            "mensagem": "Leitura OMR realizada com sucesso via OpenCV!"
        }

    except ValueError as ve:
        return {"status": "ERRO", "mensagem": str(ve)}
    except Exception as e:
        return {"status": "ERRO", "mensagem": f"Falha interna no processamento: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)