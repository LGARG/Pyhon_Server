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
    Examina o cartão de respostas detetando dinamicamente os contornos das bolhas 
    na folha alinhada, garantindo robustez a pequenas variações de impressão.
    """
    gray = cv2.cvtColor(img_alinhada, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Encontra todos os contornos na folha alinhada
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidatos_bolhas = []
    
    # Filtra contornos que correspondem ao tamanho esperado das alternativas (quadrados/bolhas)
    for c in contours:
        area = cv2.contourArea(c)
        # Ajuste estes valores de área conforme o tamanho real da alternativa no warp de 1000x1414
        if 80 < area < 600: 
            x, y, w, h = cv2.boundingRect(c)
            proporcao = w / float(h) if h > 0 else 0
            
            # Garante que são formas próximas de quadradas ou circulares
            if 0.6 <= proporcao <= 1.4:
                candidatos_bolhas.append((x, y, w, h, c))

    # Se não encontrar o número esperado de alternativas (10 questões * 5 opções = 50 alternativas)
    # ou se precisar de uma abordagem híbrida por linhas, ordenamos os candidatos encontrados:
    # Ordena primariamente por Y (agrupando por linhas/altura) e secundariamente por X (esquerda para direita)
    
    # Uma estratégia robusta para 10 questões divididas em duas colunas:
    # Vamos ordenar todos os candidatos por coordenada Y para identificar as linhas de questões
    candidatos_bolhas = sorted(candidatos_bolhas, key=lambda b: b[1])

    respostas_detectadas = {}
    acertos = 0
    
    # Agrupa por proximidade vertical (linhas aproximadas)
    linhas = []
    linha_atual = []
    y_ultimo = -1
    
    for b in candidatos_bolhas:
        x, y, w, h, c = b
        if y_ultimo == -1 or abs(y - y_ultimo) < 15:  # Tolerância vertical para a mesma linha
            linha_atual.append(b)
        else:
            if linha_atual:
                # Ordena a linha da esquerda para a direita (por X)
                linhas.append(sorted(linha_atual, key=lambda item: item[0]))
            linha_atual = [b]
        y_ultimo = y
        
    if linha_atual:
        linhas.append(sorted(linha_atual, key=lambda item: item[0]))

    # Filtra apenas linhas que contêm aglomerados de alternativas válidas 
    # (por exemplo, linhas com 5 alternativas)
    questoes_validas = [l for l in linhas if len(l) >= 5]

    # Processa cada questão detetada dinamicamente
    for idx, linha in enumerate(questoes_validas[:total_questoes]):
        q_num = str(idx + 1)
        marcatura_maxima = 0
        opcao_marcada = None

        for alt_idx, (x, y, w, h, c) in enumerate(linha[:5]):  # 5 alternativas: A, B, C, D, E
            # Cria uma máscara para calcular a densidade de pixels pretos dentro da caixa detetada
            mascara = np.zeros(thresh.shape, dtype="uint8")
            cv2.rectangle(mascara, (x, y), (x + w, y + h), 255, -1)
            
            pixels_preenchidos = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mascara))
            
            if pixels_preenchidos > marcatura_maxima:
                marcatura_maxima = pixels_preenchidos
                opcao_marcada = MAPA_LETRAS[alt_idx]

        # Define um limiar de preenchimento para considerar a alternativa marcada
        if marcatura_maxima > 100:  # Ajustável conforme o preenchimento a caneta
            respostas_detectadas[q_num] = opcao_marcada
            if gabarito_dict.get(q_num) == opcao_marcada:
                acertos += 1
        else:
            respostas_detectadas[q_num] = "N/A"

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