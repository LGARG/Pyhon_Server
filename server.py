import base64
import json
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Servidor OMR - Centralize Docente (Padrão Evalbee)")

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
    Localiza os marcadores fiduciais (quadrados pretos nos cantos) 
    para endireitar e normalizar a perspetiva da folha de respostas.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    h_img, w_img = thresh.shape
    
    # Divide a imagem em 4 quadrantes para procurar os marcadores dos cantos
    quadrantes = [
        ("TL", thresh[0:int(h_img*0.35), 0:int(w_img*0.35)]),
        ("TR", thresh[0:int(h_img*0.35), int(w_img*0.65):w_img]),
        ("BR", thresh[int(h_img*0.65):h_img, int(w_img*0.65):w_img]),
        ("BL", thresh[int(h_img*0.65):h_img, 0:int(w_img*0.35)])
    ]
    
    pontos = []
    for nome, quad in quadrantes:
        contours, _ = cv2.findContours(quad, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError(f"Marcador fiducial não encontrado no quadrante {nome}. Enquadre melhor a folha.")
        
        # Pega o maior contorno no quadrante (o quadrado preto do marcador)
        maior_c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(maior_c) < 15:
            raise ValueError(f"Marcador muito pequeno no quadrante {nome}.")
            
        M = cv2.moments(maior_c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            x, y, w, h = cv2.boundingRect(maior_c)
            cx, cy = x + w // 2, y + h // 2
            
        # Ajusta o offset baseado no quadrante analisado
        if nome == "TR":
            cx += int(w_img * 0.65)
        elif nome == "BR":
            cx += int(w_img * 0.65)
            cy += int(h_img * 0.65)
        elif nome == "BL":
            cy += int(h_img * 0.65)
            
        pontos.append([float(cx), float(cy)])

    pts = np.array(pontos, dtype="float32")
    
    # Dimensões padronizadas para o alinhamento da folha (Proporção A4)
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
    Examina o cartão aplicando correção de iluminação (antissombras) 
    e mapeamento ancorado por colunas e linhas da folha padronizada.
    """
    gray = cv2.cvtColor(img_alinhada, cv2.COLOR_BGR2GRAY)

    # Correção de iluminação de fundo (Neutraliza sombras de mãos ou ambiente)
    fundo = cv2.GaussianBlur(gray, (51, 51), 0)
    normalizada = cv2.divide(gray, fundo, scale=255)
    
    blur = cv2.GaussianBlur(normalizada, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    altura, largura = img_alinhada.shape[:2]
    acertos = 0
    respostas_detectadas = {}

    # O template possui duas colunas: Questões 1 a 5 (Esquerda) e 6 a 10 (Direita)
    for q in range(1, total_questoes + 1):
        coluna = 0 if q <= 5 else 1
        linha_index = (q - 1) % 5

        # Coordenadas proporcionais calibradas para o template
        x_inicio_base = 0.22 if coluna == 0 else 0.62
        y_inicio_base = 0.38 + (linha_index * 0.085)

        marcatura_maxima = 0
        opcao_marcada = None

        for alt_idx in range(5):  # Alternativas A, B, C, D, E
            x_bolha = int((x_inicio_base + (alt_idx * 0.052)) * largura)
            y_bolha = int(y_inicio_base * altura)
            r = int(0.014 * largura)  # Raio proporcional da bolha

            mascara = np.zeros(thresh.shape, dtype="uint8")
            cv2.circle(mascara, (x_bolha, y_bolha), r, 255, -1)
            
            pixels_preenchidos = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mascara))

            if pixels_preenchidos > marcatura_maxima:
                marcatura_maxima = pixels_preenchidos
                opcao_marcada = MAPA_LETRAS[alt_idx]

        # Limiar de tinta rigoroso (exige ao menos 30% da bolha preenchida)
        area_minima = (np.pi * (r ** 2)) * 0.30
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
        # 1. Decodifica a imagem Base64 enviada pelo Flutter
        bytes_imagem = base64.b64decode(payload.imagemBase64)
        nparr = np.frombuffer(bytes_imagem, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Imagem inválida ou corrompida.")

        # 2. Converte o gabarito JSON em dicionário
        gabarito = json.loads(payload.gabaritoOficial)

        # 3. Alinha a imagem com base nos marcadores fiduciais
        img_alinhada = alinhar_e_recortar_folha(img)

        # 4. Avalia as questões com correção de sombras e grelha fixa
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
            "mensagem": "Leitura OMR realizada com sucesso!"
        }

    except ValueError as ve:
        return {"status": "ERRO", "mensagem": str(ve)}
    except Exception as e:
        return {"status": "ERRO", "mensagem": f"Falha interna no processamento: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)