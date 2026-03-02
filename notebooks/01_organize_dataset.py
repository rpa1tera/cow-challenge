import os
import shutil
import json
from pathlib import Path

# Config Paths
BASE_DIR = Path("g:/PYTHON/cow")
TURMA_DIR = BASE_DIR / "data" / "turma"
RAW_IMG_DIR = BASE_DIR / "data" / "raw_images"
JSON_DIR = BASE_DIR / "data" / "jsons_ls"

# Create target dirs if they don't exist
RAW_IMG_DIR.mkdir(parents=True, exist_ok=True)
JSON_DIR.mkdir(parents=True, exist_ok=True)

def move_images():
    print("=== ETAPA 1: EXTRAINDO IMAGENS ===")
    moved = 0
    skipped = 0
    # Walk through TURMA_DIR
    for root, dirs, files in os.walk(TURMA_DIR):
        root_path = Path(root)
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                src_path = root_path / file
                dst_path = RAW_IMG_DIR / file
                
                if not dst_path.exists():
                    shutil.move(str(src_path), str(dst_path))
                    moved += 1
                else:
                    skipped += 1
    print(f"-> Imagens movidas para raw_images: {moved}")
    if skipped > 0:
         print(f"-> Imagens ignoradas (já existiam no destino): {skipped}")

def find_local_image_name(ls_filename):
    # 1. Exact match
    if (RAW_IMG_DIR / ls_filename).exists():
        return RAW_IMG_DIR / ls_filename
    
    # 2. Hash removed
    parts = ls_filename.split('-', 1)
    if len(parts) > 1:
        candidate = parts[1]
        if (RAW_IMG_DIR / candidate).exists():
            return RAW_IMG_DIR / candidate
            
    # 3. Fallback globally
    for p in RAW_IMG_DIR.glob("*"):
        if ls_filename.endswith(p.name) or p.name in ls_filename:
            return p
            
    return None

def extract_image_path_from_json(data):
    if isinstance(data, list) and len(data) > 0:
        data = data[0]
        
    paths_to_try = [
        lambda d: d.get("task", {}).get("data", {}).get("img"),
        lambda d: d.get("task", {}).get("data", {}).get("image"),
        lambda d: d.get("data", {}).get("img"),
        lambda d: d.get("data", {}).get("image"),
        lambda d: d.get("task", {}).get("data", {}).get("image_url"),
    ]
    
    for extractor in paths_to_try:
        try:
            res = extractor(data)
            if res:
                return res
        except:
            pass
            
    # If not found, try robust deep search
    def find_img(d):
        if isinstance(d, dict):
            for k, v in d.items():
                if k in ['img', 'image', 'image_url'] and isinstance(v, str) and (v.endswith('.jpg') or v.endswith('.png')):
                    return v
                res = find_img(v)
                if res: return res
        elif isinstance(d, list):
            for item in d:
                res = find_img(item)
                if res: return res
        return None
        
    return find_img(data)

def move_and_rename_jsons():
    print("\n=== ETAPA 2: EXTRAINDO E RENOMEANDO ANOTAÇÕES ===")
    renamed_count = 0
    errors = 0
    
    for root, dirs, files in os.walk(TURMA_DIR):
        root_path = Path(root)
        if root_path.name.lower() in ("key_points", "key_points_2", "key_points_3", "kp"):
            for file in files:
                src_path = root_path / file
                
                if file.lower().endswith(('.zip', '.rar')):
                    continue
                    
                try:
                    with open(src_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    img_path_ls = extract_image_path_from_json(data)
                    
                    if not img_path_ls:
                        continue
                        
                    img_name_ls = os.path.basename(img_path_ls)
                    local_img = find_local_image_name(img_name_ls)
                    
                    if local_img:
                        new_name = local_img.stem + ".json"
                        new_path = JSON_DIR / new_name
                        
                        shutil.move(str(src_path), str(new_path))
                        renamed_count += 1
                    else:
                        errors += 1
                        print(f"Atenção: A imagem {img_name_ls} citada no rótulo '{file}' de '{root_path.parent.name}' não foi encontrada em 'raw_images'.")
                        
                except json.JSONDecodeError:
                    pass # Not a json file
                except Exception as e:
                    errors += 1
                    
    print(f"-> Rótulos (JSONs) recuperados, renomeados e movidos: {renamed_count}")
    print(f"-> Rótulos orfãos/erros (imagem perdida): {errors}")

if __name__ == "__main__":
    move_images()
    move_and_rename_jsons()
