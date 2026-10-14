import base64
import json
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Servidor OMR - Centralize Docente (Padrão Evalbee - Alta Sensibilidade)")

MAPA_LETRAS = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}

class PayloadOMR(BaseModel):
    idCurso: str
    disciplina: str
    numeroQuestoes: int
    pontuacaoPorQuestao: float
    gabaritoOficial: str
    imagemBase64: str

def alinhar_e_recortar_folha(img):
    """
    Localiza os 6 marcadores fiduciais (cantos e laterais) com filtros 
    adaptativos para eliminar sombras e ruídos de iluminação.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Correção adaptativa de iluminação para remover sombras fortes
    fundo = cv2.GaussianBlur(gray, (51, 51), 0)
    normalizada = cv2.divide(gray, fundo, scale=255)
    
    # 2. Binarização Otsu otimizada para destacar os quadrados pretos
    blur = cv2.GaussianBlur(normalizada, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    h_img, w_img = thresh.shape
    
    # Divisão em 6 zonas correspondentes aos 6 marcadores (Top-Esq, Top-Dir, Meio-Esq, Meio-Dir, Bot-Esq, Bot-Dir)
    zonas = [
        ("TL", thresh[0:int(h_img*0.3), 0:int(w_img*0.35)]),
        ("TR", thresh[0:int(h_img*0.3), int(w_img*0.65):w_img]),
        ("ML", thresh[int(h_img*0.3):int(h_img*0.7), 0:int(w_img*0.35)]),
        ("MR", thresh[int(h_img*0.3):int(h_img*0.7), int(w_img*0.65):w_img]),
        ("BL", thresh[int(h_img*0.7):h_img, 0:int(w_img*0.35)]),
        ("BR", thresh[int(h_img*0.7):h_img, int(w_img*0.65):w_img])
    ]
    
    pontos_cantos = []
    
    for nome, zona in zonas:
        contours, _ = cv2.findContours(zona, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError(f"Marcador fiducial não detetado na zona {nome}. Verifique o enquadramento.")
        
        # Filtra contornos por área mínima para evitar ruído de texto ou pontos falsos
        contornos_validos = [c for c in contours if cv2.contourArea(c) > 10]
        if not contornos_validos:
            raise ValueError(f"Marcadores muito pequenos ou ausentes na zona {nome}.")
            
        maior_c = max(contornos_validos, key=cv2.contourArea)
            
        M = cv2.moments(maior_c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            x, y, w, h = cv2.boundingRect(maior_c)
            cx, cy = x + w // 2, y + h // 2
            
        # Reajusta o offset global de acordo com a zona analisada
        if nome in ["TR", "MR", "BR"]:
            cx += int(w_img * 0.65)
        if nome in ["ML", "MR"]:
            cy += int(h_img*0.3)
        elif nome in ["BL", "BR"]:
            cy += int(h_img * 0.7)
            
        # Guardamos primariamente os 4 cantos extremos para a perspetiva principal
        if nome in ["TL", "TR", "BR", "BL"]:
            pontos_cantos.append([float(cx), float(cy)])

    pts = np.array(pontos_cantos, dtype="float32")
    
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
    gray = cv2.cvtColor(img_alinhada, cv2.COLOR_BGR2GRAY)

    # Neutralização avançada de sombras e iluminação irregular
    fundo = cv2.GaussianBlur(gray, (51, 51), 0)
    normalizada = cv2.divide(gray, fundo, scale=255)
    
    blur = cv2.GaussianBlur(normalizada, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    altura, largura = img_alinhada.shape[:2]
    acertos = 0
    respostas_detectadas = {}

    for q in range(1, total_questoes + 1):
        coluna = 0 if q <= 5 else 1
        linha_index = (q - 1) % 5

        x_inicio_base = 0.22 if coluna == 0 else 0.62
        y_inicio_base = 0.38 + (linha_index * 0.085)

        marcatura_maxima = 0
        opcao_marcada = None

        for alt_idx in range(5):
            x_bolha = int((x_inicio_base + (alt_idx * 0.052)) * largura)
            y_bolha = int(y_inicio_base * altura)
            r = int(0.014 * largura)

            mascara = np.zeros(thresh.shape, dtype="uint8")
            cv2.circle(mascara, (x_bolha, y_bolha), r, 255, -1)
            
            pixels_preenchidos = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mascara))

            if pixels_preenchidos > marcatura_maxima:
                marcatura_maxima = pixels_preenchidos
                opcao_marcada = MAPA_LETRAS[alt_idx]

        # Limiar de preenchimento rigoroso (25% da área da bolha)
        area_minima = (np.pi * (r ** 2)) * 0.25
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
        bytes_imagem = base64.b64decode(payload.imagemBase64)
        nparr = np.frombuffer(bytes_imagem, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Imagem inválida ou corrompida.")

        gabarito = json.loads(payload.gabaritoOficial)
        img_alinhada = alinhar_e_recortar_folha(img)
        acertos, respostas = avaliar_questoes_omr(img_alinhada, payload.numeroQuestoes, gabarito)

        nota_final = acertos * payload.pontuacaoPorQuestao

        return {
            "status": "SUCESSO",
            "acertos": acertos,
            "notaFinal": round(nota_final, 2),
            "respostasDetectadas": respostas,
            "mensagem": "Leitura OMR por visão computacional concluída!"
        }

    except ValueError as ve:
        return {"status": "ERRO", "mensagem": str(ve)}
    except Exception as e:
        return {"status": "ERRO", "mensagem": f"Falha interna no processamento: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)