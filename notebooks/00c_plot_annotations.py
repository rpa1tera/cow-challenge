import os
import json
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

DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# Gera uma cor visivel para cada keypoint
def get_color(idx, total):
    hue = idx / total
    rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    return (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

COLORS = {k: get_color(i, len(KEYPOINTS)) for i, k in enumerate(KEYPOINTS)}

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

def plot_annotations():
    print(f"Gerando imagens de debug a partir dos JSONs...")
    
    json_files = list(JSON_DIR.glob("*.json"))
    if not json_files: json_files = list(JSON_DIR.glob("*"))
    
    success_count = 0
    error_count = 0
    
    for jf in json_files:
        try:
            with open(jf, 'r') as f:
                data = json.load(f)
                
            img_path_ls = data.get("task", {}).get("data", {}).get("img")
            if not img_path_ls: continue
            
            img_name_ls = os.path.basename(img_path_ls)
            local_img_path = find_local_image(img_name_ls)
            
            if not local_img_path:
                print(f"[-] Ops. Não achou raw image para {img_name_ls}")
                error_count += 1
                continue
                
            results = data.get("result", [])
            if not results and data.get("annotations"):
                results = data["annotations"][0].get("result", [])
                
            if not results: continue

            # Carrega Imagem
            img = cv2.imread(str(local_img_path))
            if img is None: continue
            
            h, w = img.shape[:2]
            
            # Percorre resultados
            for res in results:
                val = res.get("value", {})
                
                x_norm = val.get("x", 0) / 100.0
                y_norm = val.get("y", 0) / 100.0
                w_norm = val.get("width", 0) / 100.0
                h_norm = val.get("height", 0) / 100.0
                
                # Plot BBox
                if "rectanglelabels" in val:
                    rx = int(x_norm * w)
                    ry = int(y_norm * h)
                    rw = int(w_norm * w)
                    rh = int(h_norm * h)
                    cv2.rectangle(img, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)
                    cv2.putText(img, "Cow", (rx, ry-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Plot Keypoint
                if "keypointlabels" in val:
                    lbls = val.get("keypointlabels", [])
                    if not lbls: continue
                    k_name = lbls[0]
                    
                    if k_name in KEYPOINTS:
                        cx = int((x_norm + w_norm/2) * w)
                        cy = int((y_norm + h_norm/2) * h)
                        
                        color = COLORS[k_name]
                        # Desenha a bolinha e o nome flutuante ao lado
                        cv2.circle(img, (cx, cy), 6, color, -1)
                        cv2.putText(img, k_name, (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            
            # Salvar
            out_path = DEBUG_DIR / f"debug_{local_img_path.name}"
            cv2.imwrite(str(out_path), img)
            success_count += 1
            
        except Exception as e:
            print(f"Falha ao plotar {jf.name}: {e}")
            error_count += 1
            
    print(f"\nConcluído!")
    print(f"Sucessos: {success_count} fotos desenhadas na pasta: {DEBUG_DIR}")
    print(f"Erros: {error_count}")

if __name__ == "__main__":
    plot_annotations()
