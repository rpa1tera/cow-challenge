
import json
import os
import shutil
import random
from pathlib import Path

# Configuração
KEYPOINTS = [
    "withers",
    "back",
    "hook up",
    "hook down",
    "hip",
    "tail head",
    "pin up",
    "pin down"
]
KP_MAP = {k: i for i, k in enumerate(KEYPOINTS)}

# Caminhos
BASE_DIR = Path("g:/PYTHON/cow")
JSON_DIR = BASE_DIR / "data/jsons_ls"
RAW_IMG_DIR = BASE_DIR / "data/raw_images"
PROCESSED_DIR = BASE_DIR / "data/processed"

# Configurar Diretórios de Saída
DIRS = {
    "train_img": PROCESSED_DIR / "images/train",
    "val_img": PROCESSED_DIR / "images/val",
    "test_img": PROCESSED_DIR / "images/test",
    "train_lbl": PROCESSED_DIR / "labels/train",
    "val_lbl": PROCESSED_DIR / "labels/val",
    "test_lbl": PROCESSED_DIR / "labels/test",
}

for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

def find_local_image(ls_filename):
    """
    O Label Studio adiciona um prefixo de hash (ex: '570349e8-').
    Nós combinamos verificando se o arquivo local termina com as partes do nome do arquivo ls.
    """
    if (RAW_IMG_DIR / ls_filename).exists():
        return RAW_IMG_DIR / ls_filename
    
    parts = ls_filename.split('-', 1)
    if len(parts) > 1:
        candidate = parts[1]
        if (RAW_IMG_DIR / candidate).exists():
            return RAW_IMG_DIR / candidate
            
    # Fallback: procurar por arquivo que termina com o mesmo sufixo
    # Otimização: listar arquivos uma vez
    raw_files = list(RAW_IMG_DIR.glob("*"))
    for p in raw_files:
        if ls_filename.endswith(p.name) or p.name in ls_filename:
            return p
            
    return None

def convert_json_to_yolo():
    json_files = list(JSON_DIR.glob("*"))
    print(f"Found {len(json_files)} files.")
    
    data_samples = []

    for jf in json_files:
        with open(jf, 'r') as f:
            data = json.load(f)
        
        img_path_ls = data.get("task", {}).get("data", {}).get("img")
        if not img_path_ls:
            print(f"Skipping {jf}, no image path found.")
            continue
            
        img_name_ls = os.path.basename(img_path_ls)
        local_img_path = find_local_image(img_name_ls)
        
        if not local_img_path:
            print(f"Could not find local image for {img_name_ls}")
            continue
            
        results = data.get("result", [])
        
        # Precisamos encontrar os "rectanglelabels" para a VACA (COW)
        bbox = None
        keypoints = {} # "name": [x, y] (normalizado 0-1)
        
        for res in results:
            val = res.get("value", {})
            
            x_norm = val.get("x", 0) / 100.0
            y_norm = val.get("y", 0) / 100.0
            w_norm = val.get("width", 0) / 100.0
            h_norm = val.get("height", 0) / 100.0
            
            if "rectanglelabels" in val:
                cx = x_norm + (w_norm / 2)
                cy = y_norm + (h_norm / 2)
                bbox = [0, cx, cy, w_norm, h_norm]
                
            elif "keypointlabels" in val:
                lbls = val.get("keypointlabels", [])
                if not lbls: continue
                k_name = lbls[0]
                
                kx = x_norm + (w_norm / 2)
                ky = y_norm + (h_norm / 2)
                
                if k_name in KP_MAP:
                    keypoints[k_name] = [kx, ky]
        
        if bbox is None:
            if keypoints:
                xs = [p[0] for p in keypoints.values()]
                ys = [p[1] for p in keypoints.values()]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                
                pad = 0.05
                min_x = max(0, min_x - pad)
                min_y = max(0, min_y - pad)
                max_x = min(1, max_x + pad)
                max_y = min(1, max_y + pad)
                
                w = max_x - min_x
                h = max_y - min_y
                cx = min_x + w/2
                cy = min_y + h/2
                bbox = [0, cx, cy, w, h]
            else:
                print(f"Sem bbox ou keypoints para {jf}, pulando")
                continue

        kps_list = []
        for kp_name in KEYPOINTS:
            if kp_name in keypoints:
                kps_list.extend([keypoints[kp_name][0], keypoints[kp_name][1], 2]) 
            else:
                kps_list.extend([0.0, 0.0, 0])
                
        yolo_line = bbox + kps_list
        yolo_str = " ".join([f"{x:.6f}" if isinstance(x, float) else str(x) for x in yolo_line])
        
        data_samples.append({
            "img_path": local_img_path,
            "yolo_str": yolo_str
        })

    if not data_samples:
        print("Nenhuma amostra válida encontrada.")
        return

    # Divisão aleatória 70/15/15
    random.seed(42)
    random.shuffle(data_samples)
    total = len(data_samples)
    train_idx = int(total * 0.70)
    val_idx = int(total * 0.85)

    train_set = data_samples[:train_idx]
    val_set = data_samples[train_idx:val_idx]
    test_set = data_samples[val_idx:]
    
    def save_set(dataset, img_dst_dir, lbl_dst_dir):
        for item in dataset:
            src_img = item["img_path"]
            dst_img = img_dst_dir / src_img.name
            shutil.copy(src_img, dst_img)
            
            txt_name = src_img.stem + ".txt"
            dst_txt = lbl_dst_dir / txt_name
            with open(dst_txt, "w") as f:
                f.write(item["yolo_str"] + "\n")
                
    save_set(train_set, DIRS["train_img"], DIRS["train_lbl"])
    save_set(val_set, DIRS["val_img"], DIRS["val_lbl"])
    save_set(test_set, DIRS["test_img"], DIRS["test_lbl"])
    
    print(f"Processed {len(data_samples)} images.")
    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

if __name__ == "__main__":
    convert_json_to_yolo()
