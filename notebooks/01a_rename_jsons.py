
import json
import os
import shutil
from pathlib import Path

# Paths
BASE_DIR = Path("g:/PYTHON/cow")
JSON_DIR = BASE_DIR / "data/jsons_ls"
RAW_IMG_DIR = BASE_DIR / "data/raw_images"

def find_local_image_name(ls_filename):
    """
    Tenta encontrar o nome do arquivo local correspondente ao nome do Label Studio.
    Retorna o Path do arquivo local se encontrado, ou None.
    """
    # 1. Verifica correspondência exata na pasta raw
    if (RAW_IMG_DIR / ls_filename).exists():
        return RAW_IMG_DIR / ls_filename
    
    # 2. Verifica removendo prefixo de hash (ex: 'uuid-nome.jpg')
    parts = ls_filename.split('-', 1)
    if len(parts) > 1:
        candidate = parts[1]
        if (RAW_IMG_DIR / candidate).exists():
            return RAW_IMG_DIR / candidate
            
    # 3. Fallback: procura arquivo que termina com o nome
    # Isso é útil se o hash tiver hifens ou formato diferente
    raw_files = list(RAW_IMG_DIR.glob("*"))
    for p in raw_files:
        if ls_filename.endswith(p.name) or p.name in ls_filename:
            return p
            
    return None

def rename_jsons():
    print(f"Scanning {JSON_DIR}...")
    # Files might not have .json extension (e.g. '1', '100')
    json_files = list(JSON_DIR.glob("*"))
    
    renamed_count = 0
    errors = 0
    
    for jf in json_files:
        try:
            with open(jf, 'r') as f:
                data = json.load(f)
            
            # Extrair caminho da imagem do JSON
            img_path_ls = data.get("task", {}).get("data", {}).get("img")
            if not img_path_ls:
                print(f"Ignorando {jf.name}: Caminho da imagem não encontrado no JSON.")
                continue
                
            img_name_ls = os.path.basename(img_path_ls)
            
            # Encontrar arquivo de imagem real correspondente
            local_img = find_local_image_name(img_name_ls)
            
            if local_img:
                # Novo nome do JSON = nome_da_imagem + .json
                # Ex: imagem.jpg -> imagem.json
                new_name = local_img.stem + ".json"
                new_path = JSON_DIR / new_name
                
                if new_path != jf:
                    # Renomear
                    print(f"Renomeando: {jf.name} -> {new_name}")
                    # Usar shutil.move para renomear
                    shutil.move(str(jf), str(new_path))
                    renamed_count += 1
                else:
                    # Já está correto
                    pass
            else:
                print(f"Não foi possível encontrar imagem local para {jf.name} (referência: {img_name_ls})")
                errors += 1
                
        except Exception as e:
            print(f"Erro ao processar {jf.name}: {e}")
            errors += 1

    print(f"Concluído. Renomeados: {renamed_count}, Erros/Não encontrados: {errors}")

if __name__ == "__main__":
    rename_jsons()
