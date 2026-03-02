import os
import cv2
import numpy as np
# Desativar integrações via variáveis de ambiente simples
os.environ["WANDB_DISABLED"] = "true"
os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"

from ultralytics import YOLO, settings
import math
import csv
from pathlib import Path
import itertools

# Desativar explicitamente via API de configurações
try:
    settings.update({"mlflow": False, "wandb": False, "clearml": False, "comet": False, "tensorboard": False})
except Exception as e:
    pass

# Verificação do mapeamento de Keypoints
KEYPOINTS = [
    "withers", "back", "hook up", "hook down", "hip", "tail head", "pin up", "pin down"
]

def distance(p1, p2):
    if p1 is None or p2 is None: return None
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_angle(p1, p2, p3):
    # Retorna o ângulo no vértice p2 formado entre (p1, p2, p3)
    if p1 is None or p2 is None or p3 is None: return None
    a = distance(p2, p3)
    b = distance(p1, p3)
    c = distance(p1, p2)
    
    if a == 0 or c == 0: return None
    
    # Lei dos Cossenos
    v = (a**2 + c**2 - b**2) / (2 * a * c)
    v = max(-1.0, min(1.0, v)) # Segurança contra erros de ponto flutuante
    return math.degrees(math.acos(v))

def polygon_area(points):
    # Fórmula do Topógrafo (Shoelace Formula)
    if any(p is None for p in points): return None
    n = len(points)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    return abs(area) / 2.0

def extract_features(mode="both"):
    model_path = 'g:/PYTHON/cow/models/yolov26n-cow-pose/weights/best.pt'
    
    if not os.path.exists(model_path):
        print(f"Modelo não encontrado em {model_path}.")
        return

    model = YOLO(model_path)
    dataset_dir = Path('g:/PYTHON/cow/data/dataset_classificação')
    
    if mode == "geo":
        output_csv = 'g:/PYTHON/cow/data/processed/cow_features_geo.csv'
    else:
        output_csv = 'g:/PYTHON/cow/data/processed/cow_features_hybrid.csv'
    
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    
    # Estratégias EXATAS do Estudo 1: 16 Distâncias Específicas
    # Mapeamento do Paper (Figure 1):
    # 1: hook down (Hip Right)
    # 2: hook up (Hip Left)
    # 3: pin down (Pin Right)
    # 4: pin up (Pin Left)
    # 5: tail head
    # 6: back (Sacral)
    # 7: withers (Cervical)
    
    f_ra_pairs = [
        ("hook down", "hook up"),    # F1: 1 -> 2
        ("pin down", "pin up"),      # F2: 3 -> 4
        ("tail head", "back"),       # F3: 5 -> 6
        ("hook down", "pin down"),   # F4: 1 -> 3
        ("hook up", "pin up"),       # F5: 2 -> 4
        ("pin down", "back"),        # F6: 3 -> 6
        ("pin up", "back"),          # F7: 4 -> 6
        ("pin down", "tail head"),   # F8: 3 -> 5
        ("pin up", "tail head"),     # F9: 4 -> 5
        ("hook down", "back"),       # F10: 1 -> 6
        ("hook up", "back"),         # F11: 2 -> 6
        ("hook down", "tail head"),  # F12: 1 -> 5
        ("hook up", "tail head"),    # F13: 2 -> 5
    ]
    
    f_da_pairs = [
        ("back", "withers"),         # F14: 6 -> 7
        ("hook down", "withers"),    # F15: 1 -> 7
        ("hook up", "withers"),      # F16: 2 -> 7
    ]
    
    s1_cols = [f"S1_F{i}" for i in range(1, 14)]
    s2_cols = [f"S2_F{i}" for i in range(1, 17)]
    img_cols = ["ratio_white_fur", "ratio_black_fur", "texture_contrast"]
    
    header = ["filename", "cow_id"] + s1_cols + s2_cols
    if mode == "both":
        header += img_cols
    
    images = list(dataset_dir.glob("*/*"))
    print(f"Extraindo características avançadas (Biometria) de {len(images)} imagens...")
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        for img_path in images:
            if not img_path.is_file():
                continue
                
            results = model(str(img_path), verbose=False)
            
            for r in results:
                if r.keypoints is None or len(r.keypoints.xyn) == 0:
                    continue
                
                # Vamos pegar pixels da imagem XY para não ficarmos refém da normalização cega da bounding box.
                kps = r.keypoints.xy[0].tolist()
                conf = r.keypoints.conf[0].tolist()
                
                def get_k(idx):
                    # Como o YOLO26 pode ter calibração de confiança numérica menor,
                    # reduzimos de 0.5 para 0.25 para não perder os pontos vitais.
                    if conf[idx] < 0.25: return None
                    return kps[idx]

                kp_map = {name: get_k(i) for i, name in enumerate(KEYPOINTS)}
                
                # 1. Distâncias da Garupa (Rump Area - Strategy 1) F1 a F13
                dists_f1_13 = [distance(kp_map[p[0]], kp_map[p[1]]) for p in f_ra_pairs]
                valid_ra = [d for d in dists_f1_13 if d is not None]
                sum_ra = sum(valid_ra) if len(valid_ra) == 13 else 0
                
                # 2. Distâncias do Dorso Total (Dorsal Area - Strategy 2 complement) F14 a F16
                dists_f14_16 = [distance(kp_map[p[0]], kp_map[p[1]]) for p in f_da_pairs]
                
                # Todas as distâncias para a soma global (Strategy 2)
                all_dists = dists_f1_13 + dists_f14_16
                valid_all = [d for d in all_dists if d is not None]
                sum_all = sum(valid_all) if len(valid_all) == 16 else 0
                
                if sum_ra == 0 or sum_all == 0: continue
                
                # Scale Strategy 1 (por cento da garupa): divide puramente F1 a F13 pela soma deles
                strat1_feats = [(d / sum_ra) for d in dists_f1_13]
                
                # Scale Strategy 2: divide puramente F1 a F16 pela soma global (1-16)
                strat2_feats = [(d / sum_all) for d in all_dists]
                
                img_feats = []
                if mode == "both":
                    # ==========================================
                    # HIBRIDIZAÇÃO: Extração de Padrão de Pelagem Visual
                    # ==========================================
                    ratio_white, ratio_black, texture_contrast = 0.0, 0.0, 0.0
                    if hasattr(r, 'boxes') and r.boxes is not None and len(r.boxes.xyxy) > 0:
                        box = r.boxes.xyxy[0].tolist()
                        x1, y1, x2, y2 = map(int, box)
                        
                        orig_img = r.orig_img
                        h_img, w_img = orig_img.shape[:2]
                        
                        # Garantir que a bounding box esteja dentro da imagem
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w_img, x2), min(h_img, y2)
                        
                        if x2 > x1 and y2 > y1:
                            # Extrai SOMENTE a vaca (Bounding Box isolada)
                            crop = orig_img[y1:y2, x1:x2]
                            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                            
                            # 1. Proporção Global da Pelagem (Limiarização de Otsu/Básica)
                            # Assumindo 127 como limiar médio de iluminação pra separar mancha preta de pele branca
                            _, binary = cv2.threshold(gray_crop, 127, 255, cv2.THRESH_BINARY)
                            total_p = binary.shape[0] * binary.shape[1]
                            if total_p > 0:
                                w_pixels = cv2.countNonZero(binary)
                                b_pixels = total_p - w_pixels
                                ratio_white = w_pixels / total_p
                                ratio_black = b_pixels / total_p
                                
                            # 2. Textura Orgânica / Contraste do Pelo
                            # O desvio padrão da distribuição de tons de cinza modela se a vaca
                            # é muito "pintada/chumbada" ou se tem blocos homogêneos grandes.
                            texture_contrast = float(np.std(gray_crop))
                    
                    img_feats = [ratio_white, ratio_black, texture_contrast]
                
                cw_id = img_path.parent.name
                
                row = [img_path.name, cw_id] + strat1_feats + strat2_feats + img_feats
                writer.writerow(row)

    print(f"Características biométricas salvas em {output_csv}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extrai features das imagens para treinamento.")
    parser.add_argument("--mode", type=str, default="both", choices=["geo", "both"],
                        help="Modo de extração: 'geo' para geométricas apenas, 'both' para geométricas + pelagem (híbrido)")
    args = parser.parse_args()
    
    extract_features(mode=args.mode)
