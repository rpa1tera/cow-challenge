import os
import shutil
import json
import math
import numpy as np
import pandas as pd
import cv2
import colorsys
from pathlib import Path

# Configuração
KEYPOINTS = [
    "withers", "back", "hook up", "hook down", "hip", "tail head", "pin up", "pin down"
]

BASE_DIR = Path('g:/PYTHON/cow')
JSON_DIR = BASE_DIR / "data/jsons_ls"
RAW_IMG_DIR = BASE_DIR / "data/raw_images"
DEBUG_DIR = BASE_DIR / "data/debug_plots"
DEBUG_JSON_DIR = BASE_DIR / "data/debug_json"

DEBUG_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_JSON_DIR.mkdir(parents=True, exist_ok=True)

def get_color(idx, total):
    hue = idx / total
    rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    return (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

COLORS = {k: get_color(i, len(KEYPOINTS)) for i, k in enumerate(KEYPOINTS)}

def distance(p1, p2):
    if p1 is None or p2 is None: return None
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def extract_keypoints(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
        
    results = data.get("result", [])
    if not results and data.get("annotations"):
        results = data["annotations"][0].get("result", [])
        
    keypoints = {}
    duplicates = []
    orig_w, orig_h = None, None
    for res in results:
        val = res.get("value", {})
        if orig_w is None:
            orig_w = res.get("original_width")
            orig_h = res.get("original_height")
            
        if "keypointlabels" in val:
            lbls = val.get("keypointlabels", [])
            if lbls:
                k_name = lbls[0]
                # Label studio armazena X e Y como porcentagem (ex 84%) e largura do ponto
                x_pct = val.get("x", 0)
                y_pct = val.get("y", 0)
                if k_name in KEYPOINTS:
                    if k_name in keypoints:
                        duplicates.append(k_name)
                    keypoints[k_name] = [x_pct, y_pct]
                    
    # Converter para Pixels Reais da Imagem se tiver resolução
    if orig_w and orig_h:
        for k in keypoints:
            keypoints[k][0] = (keypoints[k][0] / 100) * orig_w
            keypoints[k][1] = (keypoints[k][1] / 100) * orig_h
                    
    return keypoints, duplicates

def find_local_image(ls_filename):
    if (RAW_IMG_DIR / ls_filename).exists():
        return RAW_IMG_DIR / ls_filename
    
    parts = ls_filename.split('-', 1)
    if len(parts) > 1:
        candidate = parts[1]
        if (RAW_IMG_DIR / candidate).exists():
            return RAW_IMG_DIR / candidate
            
    raw_files = list(RAW_IMG_DIR.glob("*"))
    for p in raw_files:
        if ls_filename.endswith(p.name) or p.name in ls_filename:
            return p
            
    return None

def plot_problematic_annotation(jf_path):
    try:
        with open(jf_path, 'r') as f:
            data = json.load(f)
            
        img_path_ls = data.get("task", {}).get("data", {}).get("img")
        if not img_path_ls: return False
        
        img_name_ls = os.path.basename(img_path_ls)
        local_img_path = find_local_image(img_name_ls)
        
        if not local_img_path: return False
            
        results = data.get("result", [])
        if not results and data.get("annotations"):
            results = data["annotations"][0].get("result", [])
            
        if not results: return False

        img = cv2.imread(str(local_img_path))
        if img is None: return False
        
        h, w = img.shape[:2]
        
        for res in results:
            val = res.get("value", {})
            x_norm = val.get("x", 0) / 100.0
            y_norm = val.get("y", 0) / 100.0
            w_norm = val.get("width", 0) / 100.0
            h_norm = val.get("height", 0) / 100.0
            
            if "rectanglelabels" in val:
                rx, ry = int(x_norm * w), int(y_norm * h)
                rw, rh = int(w_norm * w), int(h_norm * h)
                cv2.rectangle(img, (rx, ry), (rx+rw, ry+rh), (0, 0, 255), 2) # Vermelho pro erro
                cv2.putText(img, "Cow", (rx, ry-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            if "keypointlabels" in val:
                lbls = val.get("keypointlabels", [])
                if not lbls: continue
                k_name = lbls[0]
                
                if k_name in KEYPOINTS:
                    cx, cy = int((x_norm + w_norm/2) * w), int((y_norm + h_norm/2) * h)
                    color = COLORS[k_name]
                    cv2.circle(img, (cx, cy), 6, color, -1)
                    cv2.putText(img, k_name, (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        
        out_path = DEBUG_DIR / f"erro_anotacao_{local_img_path.name}"
        cv2.imwrite(str(out_path), img)
        return True
    except Exception as e:
        print(f"Falha ao plotar {jf_path.name}: {e}")
        return False

def is_left_of_line(A, B, P):
    # Retorna > 0 se o ponto P está à esquerda da linha orientada de A para B (ou seja, "acima" ou "UP" dependendo do vetor)
    # Fórmula do determinante matricial 2D
    return (B[0] - A[0]) * (P[1] - A[1]) - (B[1] - A[1]) * (P[0] - A[0])

def segments_intersect(p1, p2, p3, p4):
    # Função para checar se segmento de reta p1-p2 cruza com segmento p3-p4
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

def project_point_on_vector(A, B, P):
    # Projeta o ponto P no vetor AB, retornando um escalar (porcentagem da distância percorrida ao longo de AB)
    AB_x = B[0] - A[0]
    AB_y = B[1] - A[1]
    AP_x = P[0] - A[0]
    AP_y = P[1] - A[1]
    
    dot_product = AP_x * AB_x + AP_y * AB_y
    mag_sq = AB_x**2 + AB_y**2
    if mag_sq == 0: return 0
    return dot_product / mag_sq

def validate_geometry():
    print(f"Varrendo JSONs malucos (Erros Humanos) em: {JSON_DIR}...\n")
    json_files = list(JSON_DIR.glob("*.json"))
    if not json_files: json_files = list(JSON_DIR.glob("*"))
    
    erros_fisicos = []
    medidas_populacao = []
    arquivos_com_erro = set()
    
    for jf in json_files:
        kps, duplicates = extract_keypoints(jf)
        
        issues = []
        if duplicates:
            dups_str = ", ".join(set(duplicates))
            issues.append(f"Keypoints DUPLICADOS detectados: {dups_str}. A pessoa clicou no mesmo osso mais de uma vez!")
            
        if len(kps) < 8 and not duplicates: continue
        
        withers = kps.get("withers")
        tail_head = kps.get("tail head")
        hook_up = kps.get("hook up")
        hook_down = kps.get("hook down")
        pin_up = kps.get("pin up")
        pin_down = kps.get("pin down")
        
        if not all([withers, tail_head, hook_up, hook_down, pin_up, pin_down]):
             if issues:
                 erros_fisicos.append({"arquivo": jf.name, "erros": issues})
                 arquivos_com_erro.add(jf)
             continue
             
        comp_vaca = distance(withers, tail_head)
        largura_hooks = distance(hook_up, hook_down)
        largura_pins = distance(pin_up, pin_down)
        
        issues = []
        if largura_hooks > comp_vaca:
            issues.append("Largura dos Ossos da Bacia (Hooks) é MAIOR que a Vaca. Anotação Trocada!")
            
        if distance(withers, pin_up) < distance(tail_head, pin_up):
             issues.append("Pin Up está grudado no Pescoço (Withers) em vez do Rabo. Clicou no lugar invertido!")
             
        # --- REGRAS ALGÉBRICAS AVANÇADAS ---
        
        # NOVO: Teste de Proximidade da Coluna (Pontos sobrepostos)
        # Pontos ao longo da coluna não podem estar um "em cima do outro"
        limite_prox = comp_vaca * 0.05  # Limite de 5% do comprimento da vaca
        
        dist_w_b = distance(withers, kps.get("back"))
        dist_b_h = distance(kps.get("back"), kps.get("hip"))
        dist_h_t = distance(kps.get("hip"), tail_head)
        
        if dist_w_b is not None and dist_w_b < limite_prox:
            issues.append("Withers e Back estão quase um em cima do outro (Distância < 5% do comp).")
        if dist_b_h is not None and dist_b_h < limite_prox:
            issues.append("Back e Hip estão quase um em cima do outro (Distância < 5% do comp).")
        if dist_h_t is not None and dist_h_t < limite_prox:
            issues.append("Hip e Tail Head estão quase um em cima do outro (Distância < 5% do comp).")
            
        # 1. Teste de Progressão Longitudinal (Eixo Cervico-Caudal)
        # O Back tem que estar DEPOIS do Withers, e o Hip tem que estar DEPOIS do Back.
        proj_back = project_point_on_vector(withers, tail_head, kps.get("back", withers))
        proj_hip = project_point_on_vector(withers, tail_head, kps.get("hip", withers))
        
        if proj_back < 0:
            issues.append("Back (Meio das costas) desenhado NA FRENTE do Withers (Pescoço). Impossível.")
        if proj_hip < proj_back:
            issues.append("Hip (Meio da bacia) desenhado NA FRENTE do Back (Meio das costas). Impossível.")
            
        # 2. Teste do Cruzamento do Eixo Y da Coluna
        # 'withers' para 'tail_head' divide a vaca ao meio. 
        # 'hook up' e 'pin up' (lados UP) DEVEM estar do mesmo lado geométrico da reta da coluna.
        # 'hook down' e 'pin down' DEVEM estar do outro lado.
        cross_hook_up = is_left_of_line(withers, tail_head, hook_up)
        cross_pin_up = is_left_of_line(withers, tail_head, pin_up)
        cross_hook_down = is_left_of_line(withers, tail_head, hook_down)
        cross_pin_down = is_left_of_line(withers, tail_head, pin_down)
        
        # Sinais numéricos determinam o lado (Positivo ou Negativo). Os UP devem ter mesmo sinal do UP, os DOWN do DOWN.
        if np.sign(cross_hook_up) != np.sign(cross_pin_up) and abs(cross_hook_up) > 10 and abs(cross_pin_up) > 10:
             issues.append("O 'hook UP' está de um lado da coluna e o 'pin UP' cruzou pro outro lado da coluna. O Lado 'UP' está torto!")
             
        if np.sign(cross_hook_up) == np.sign(cross_hook_down) and abs(cross_hook_down) > 10 and abs(cross_hook_up) > 10:
            issues.append("Ambos os Ganchos (Hook UP e DOWN) estão do mesmo lado da vaca cruzando o eixo central.")
            
        # 3. Teste do Paralelismo Lateral do Quadril
        # A linha da perna direita (hook up pra pin up) não pode cruzar com a esquerda (hook down pra pin down)
        if segments_intersect(hook_up, pin_up, hook_down, pin_down):
            issues.append("Linhas do quadril Direito e Esquerdo se cruzaram (formando um X em cima da vaca). Anotador misturou Upos e Downs.")

        if issues:
            erros_fisicos.append({"arquivo": jf.name, "erros": issues})
            arquivos_com_erro.add(jf)
            
        medidas_populacao.append({
            "arquivo": jf.name,
            "path": jf,
            "dist_hook_hook": largura_hooks,
            "dist_pin_pin": largura_pins,
            "dist_tail_back": distance(tail_head, kps.get("back")),
            "dist_withers_back": distance(withers, kps.get("back"))
        })
        
    df = pd.DataFrame(medidas_populacao)
    erros_estatisticos = []
    
    if not df.empty:
        cols = [c for c in df.columns if c.startswith("dist_")]
        
        for col in cols:
            mean = df[col].mean()
            std = df[col].std()
            upper_bound = mean + (3 * std)
            lower_bound = max(0, mean - (3 * std))
            
            outliers = df[(df[col] > upper_bound) | (df[col] < lower_bound)]
            
            for _, row in outliers.iterrows():
                val = row[col]
                erros_estatisticos.append(
                    {"arquivo": row["arquivo"], "erros": [f"Medida {col} absurdamente fora do padrão (Z-score > 3): {val:.2f} (Média Ideal: {mean:.2f})"]}
                )
                arquivos_com_erro.add(row["path"])
                
    print("="*60)
    print("🚩 RELATÓRIO DE ABSURDOS ANATÔMICOS (Física)")
    print("="*60)
    for erro in erros_fisicos:
        print(f"[{erro['arquivo']}]")
        for e in erro['erros']: print(f"  ❌ {e}")
        
    print("\n" + "="*60)
    print("📊 RELATÓRIO DE MUTANTES (Outliers Estatísticos > 3 Desvios)")
    print("="*60)
    est_grouped = {}
    for est in erros_estatisticos:
        arq = est["arquivo"]
        if arq not in est_grouped: est_grouped[arq] = []
        est_grouped[arq].extend(est["erros"])
        
    for arq, errs in est_grouped.items():
        print(f"[{arq}]")
        for e in errs: print(f"  👽 {e}")
        
    print(f"\nTotal Analisado: {len(json_files)} arquivos.")
    
    print("\n" + "="*60)
    print(f"📸 GERANDO PLOTS APENAS PARA AS {len(arquivos_com_erro)} VACAS PROBLEMÁTICAS...")
    print("="*60)
    
    # Limpando plotagens antigas (filtrando apenas os plots gerados para não deletar os raws movidos)
    for old_file in DEBUG_DIR.glob("erro_anotacao_*"):
        try:
            old_file.unlink()
        except:
            pass
            
    plotted = 0
    moved_raw = 0
    moved_json = 0
    for bad_jf in arquivos_com_erro:
        if plot_problematic_annotation(bad_jf):
            plotted += 1
            
        # Mover a imagem e o JSON defeituosos ao invés de deletar permanentemente
        try:
            with open(bad_jf, 'r') as f:
                data = json.load(f)
            img_path_ls = data.get("task", {}).get("data", {}).get("img")
            
            if img_path_ls:
                img_name_ls = os.path.basename(img_path_ls)
                local_img_path = find_local_image(img_name_ls)
                # Move a RAW Image para a pasta de debug_plots
                if local_img_path and local_img_path.exists():
                    shutil.move(str(local_img_path), str(DEBUG_DIR / local_img_path.name))
                    moved_raw += 1
                    
            # Move o JSON do Label Studio para a pasta debug_json
            if bad_jf.exists():
                shutil.move(str(bad_jf), str(DEBUG_JSON_DIR / bad_jf.name))
                moved_json += 1
        except Exception as e:
            print(f"Erro ao mover arquivos defeituosos de {bad_jf.name}: {e}")
            
    print(f"Sucesso! {plotted} gabaritos visuais de vacas erradas salvos em:\n{DEBUG_DIR}")
    print(f"🧹 MANUTENÇÃO: {moved_raw} Imagens brutas movidas para debug_plots e {moved_json} anotações JSON movidas para debug_json.")

if __name__ == "__main__":
    validate_geometry()
