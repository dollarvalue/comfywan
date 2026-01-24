import modal
from pathlib import Path

# --- Configuration ---
model_vol = modal.Volume.from_name("ltx-model-storage", create_if_missing=True)
data_vol = modal.Volume.from_name("ltx-project-data", create_if_missing=True)

DATA_BASE_PATH = "/root/data"
COMFYUI_BASE_PATH = "/root/ComfyUI"
COMFYUI_MODELS_PATH = f"{COMFYUI_BASE_PATH}/models"
CHECKPOINTS_PATH = f"{COMFYUI_BASE_PATH}/models/checkpoints"
VAE_PATH = f"{COMFYUI_BASE_PATH}/models/vae"
UPSCALE_MODELS_PATH = f"{COMFYUI_BASE_PATH}/models/upscale_models"
CLIP_PATH = f"{COMFYUI_BASE_PATH}/models/clip"

app = modal.App("ltx-video-comfyui")

# --- Robust Node Downloader ---
def download_custom_nodes():
    import urllib.request
    import zipfile
    import os
    import shutil

    nodes = {
        "ComfyUI-LTXVideo": "https://github.com/Lightricks/ComfyUI-LTXVideo/archive/refs/heads/master.zip",
        "ComfyUI-VideoHelperSuite": "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite/archive/refs/heads/main.zip",
        "ComfyUI-Manager": "https://github.com/ltdrdata/ComfyUI-Manager/archive/refs/heads/main.zip",
        "ComfyUI-KJNodes": "https://github.com/kijai/ComfyUI-KJNodes/archive/refs/heads/main.zip",
        "ComfyUI_Fill-Nodes": "https://github.com/filliptm/ComfyUI_Fill-Nodes/archive/refs/heads/main.zip",
    }

    for name, url in nodes.items():
        target_path = f"/root/ComfyUI/custom_nodes/{name}"
        os.makedirs(target_path, exist_ok=True)
        zip_path = f"/tmp/{name}.zip"
        try:
            print(f"Installing {name} from {url}...")
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                temp_extract = f"/tmp/{name}_extract"
                if os.path.exists(temp_extract): shutil.rmtree(temp_extract)
                zip_ref.extractall(temp_extract)
                root_folder = os.listdir(temp_extract)[0]
                source_dir = os.path.join(temp_extract, root_folder)
                for item in os.listdir(source_dir):
                    s = os.path.join(source_dir, item)
                    d = os.path.join(target_path, item)
                    if os.path.isdir(s):
                        shutil.copytree(s, d, dirs_exist_ok=True)
                    else:
                        shutil.copy2(s, d)
            print(f"Successfully installed {name}")
        except Exception as e:
            print(f"Error installing {name}: {e}")

# --- Container Image Setup ---
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "libgl1", "libglib2.0-0", "ffmpeg", "unzip")
    .pip_install(
        "torch", "torchvision", "torchaudio", "transformers", 
        "huggingface_hub", "einops", "sentencepiece", "accelerate"
    )
    .run_commands(
        "git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /root/ComfyUI",
        "cd /root/ComfyUI && pip install -r requirements.txt",
        # Fix: Clear the directory so the Volume can mount
        "rm -rf /root/ComfyUI/models && mkdir /root/ComfyUI/models"
    )
    .run_function(download_custom_nodes)
    .run_commands(
        "pip install -r /root/ComfyUI/custom_nodes/ComfyUI-LTXVideo/requirements.txt || true",
        "pip install -r /root/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt || true",
        "pip install -r /root/ComfyUI/custom_nodes/ComfyUI-KJNodes/requirements.txt || true",
        "pip install -r /root/ComfyUI/custom_nodes/ComfyUI_Fill-Nodes/requirements.txt || true"
    )
)

MODELS_TO_DOWNLOAD = {
    "checkpoints": {
        "ltx-2-19b-phr00tmerge-nsfw-v6.safetensors": "https://huggingface.co/Phr00t/LTX2-Rapid-Merges/resolve/main/nsfw/ltx-2-19b-phr00tmerge-nsfw-v6.safetensors"
    },
    "vae": {
        "LTX2_audio_vae_bf16.safetensors": "https://huggingface.co/Kijai/LTXV2_comfy/resolve/main/VAE/LTX2_audio_vae_bf16.safetensors",
        "LTX2_video_vae_bf16.safetensors": "https://huggingface.co/Kijai/LTXV2_comfy/resolve/main/VAE/LTX2_video_vae_bf16.safetensors",
    },
    "upscale_models": {
        "ltx-2-temporal-upscaler-x2-1.0.safetensors": "https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-temporal-upscaler-x2-1.0.safetensors"
    },
    "clip": {
        "ltx-2-19b-embeddings_connector_dev_bf16.safetensors": "https://huggingface.co/Kijai/LTXV2_comfy/resolve/main/text_encoders/ltx-2-19b-embeddings_connector_dev_bf16.safetensors",
        "gemma_3_12B_it_fp8_scaled.safetensors": "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors"
    }
}

@app.function(
    image=image,
    volumes={COMFYUI_MODELS_PATH: model_vol},
    timeout=7200, 
)
def download_models():
    import os
    from pathlib import Path
    
    # Internal import to prevent local environment errors
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return

    # (Repo ID, Remote Filename, ComfyUI Subfolder)
    models = [
        ("Phr00t/LTX2-Rapid-Merges", "nsfw/ltx-2-19b-phr00tmerge-nsfw-v6.safetensors", "checkpoints"),
        ("Kijai/LTXV2_comfy", "VAE/LTX2_audio_vae_bf16.safetensors", "vae"),
        ("Kijai/LTXV2_comfy", "VAE/LTX2_video_vae_bf16.safetensors", "vae"),
        ("Lightricks/LTX-2", "ltx-2-temporal-upscaler-x2-1.0.safetensors", "upscale_models"),
        ("Kijai/LTXV2_comfy", "text_encoders/ltx-2-19b-embeddings_connector_dev_bf16.safetensors", "clip"),
        ("Comfy-Org/ltx-2", "split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors", "clip"),
    ]

    for repo_id, filename, subfolder in models:
        # Define the absolute path within the Volume
        target_dir = Path(COMFYUI_MODELS_PATH) / subfolder
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine the final local filename
        local_filename = os.path.basename(filename)
        final_path = target_dir / local_filename

        print(f"Downloading {local_filename} to {target_dir}...")

        # Download directly to the volume path
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=target_dir,
            local_dir_use_symlinks=False  # Forces actual file download into the volume
        )

    print("All models successfully stored in the volume.")
    model_vol.commit() # Commit the changes to persist them
    
@app.function(
    image=image,
    gpu="A100-80GB", 
    volumes={COMFYUI_MODELS_PATH: model_vol, DATA_BASE_PATH: data_vol},
    timeout=7200,
    min_containers=1
)
@modal.web_server(8188, startup_timeout=600) 
def start_comfyui():
    import subprocess
    import os
    
    # Ensure input/output dirs exist in data volume
    os.makedirs(os.path.join(DATA_BASE_PATH, "input"), exist_ok=True)
    os.makedirs(os.path.join(DATA_BASE_PATH, "output"), exist_ok=True)

    subprocess.Popen([
        "python", "/root/ComfyUI/main.py", 
        "--listen", "0.0.0.0", "--port", "8188",
        "--input-directory", os.path.join(DATA_BASE_PATH, "input"),
        "--output-directory", os.path.join(DATA_BASE_PATH, "output"),
        "--bf16-unet"
    ])