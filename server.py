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
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    h_img, w_img = thresh.shape
    
    # Divide a imagem em 4 quadrantes para procurar o marcador preto em cada canto
    quadrantes = [
        ("TL", thresh[0:int(h_img*0.3), 0:int(w_img*0.3)]),         # Top-Left
        ("TR", thresh[0:int(h_img*0.3), int(w_img*0.7):w_img]),     # Top-Right
        ("BR", thresh[int(h_img*0.7):h_img, int(w_img*0.7):w_img]), # Bottom-Right
        ("BL", thresh[int(h_img*0.7):h_img, 0:int(w_img*0.3)])      # Bottom-Left
    ]
    
    pontos = []
    for nome, quad in quadrantes:
        contours, _ = cv2.findContours(quad, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError(f"Marcador não encontrado no quadrante {nome}. Verifique o enquadramento.")
        
        # Pega o maior contorno no quadrante (presumivelmente o marcador fiducial)
        maior_c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(maior_c) < 20:
            raise ValueError(f"Marcador muito pequeno ou ausente no quadrante {nome}.")
            
        M = cv2.moments(maior_c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            x, y, w, h = cv2.boundingRect(maior_c)
            cx, cy = x + w // 2, y + h // 2
            
        # Ajusta o offset baseado em qual quadrante estamos
        if nome == "TR":
            cx += int(w_img * 0.7)
        elif nome == "BR":
            cx += int(w_img * 0.7)
            cy += int(h_img * 0.7)
        elif nome == "BL":
            cy += int(h_img * 0.7)
            
        pontos.append([float(cx), float(cy)])

    pts = np.array(pontos, dtype="float32")
    
    # Ordem esperada: TL, TR, BR, BL
    largura, altura = 1000, 1414
    dst = np.array([
        [0, 0],
        [largura - 1, 0],
        [largura - 1, altura - 1],
        [0, altura - 1]
    ], dtype="float32")

    M_persp = cv2.getPerspectiveTransform(pts, dst)
    warp = cv2.warpPerspective(img, M_persp, (largura, altura))

    return warp

def avaliar_questoes_omr(img_alinhada, total_questoes, gabarito_dict):
    """
    Examina o cartão aplicando correção de iluminação para anular sombras 
    e usa limiarização adaptativa para detetar apenas a tinta da caneta.
    """
    gray = cv2.cvtColor(img_alinhada, cv2.COLOR_BGR2GRAY)

    # 1. REMOÇÃO DE SOMBRAS (Correção de Iluminação de Fundo)
    # Cria uma versão desfocada da imagem para estimar o mapa de sombras/iluminação de fundo
    fundo = cv2.GaussianBlur(gray, (51, 51), 0)
    
    # Subtrai o fundo da imagem original e normaliza o contraste
    normalizada = cv2.divide(gray, fundo, scale=255)
    
    # Suaviza a imagem normalizada para eliminar ruídos de impressão
    blur = cv2.GaussianBlur(normalizada, (5, 5), 0)
    
    # 2. BINARIZAÇÃO ROBUSTA (Otsu sobre imagem livre de sombras)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    altura, largura = img_alinhada.shape[:2]
    acertos = 0
    respostas_detectadas = {}

    for q in range(1, total_questoes + 1):
        coluna = 0 if q <= 5 else 1
        linha_index = (q - 1) % 5

        x_inicio_base = 0.18 if coluna == 0 else 0.58
        y_inicio_base = 0.34 + (linha_index * 0.09)

        marcatura_maxima = 0
        opcao_marcada = None

        for alt_idx in range(5):  # A, B, C, D, E
            x_bolha = int((x_inicio_base + (alt_idx * 0.055)) * largura)
            y_bolha = int(y_inicio_base * altura)
            r = int(0.015 * largura)

            mascara = np.zeros(thresh.shape, dtype="uint8")
            cv2.circle(mascara, (x_bolha, y_bolha), r, 255, -1)
            
            pixels_preenchidos = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mascara))

            if pixels_preenchidos > marcatura_maxima:
                marcatura_maxima = pixels_preenchidos
                opcao_marcada = MAPA_LETRAS[alt_idx]

        # Limiar de preenchimento rigoroso para ignorar sombras residuais
        area_bolha = np.pi * (r ** 2)
        area_minima = area_bolha * 0.35  # Exige pelo menos 35% da bolha preenchida com tinta
        
        if marcatura_maxima >= area_minima:
            respostas_detectadas[str(q)] = opcao_marcada
            if str(gabarito_dict.get(str(q))).strip().upper() == str(opcao_marcada).strip().upper():
                acertos += 1
        else:
            respostas_detectadas[str(q)] = "N/A"

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