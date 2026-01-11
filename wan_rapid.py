import modal
import os

# 1. Updated Image definition
# 1. Image definition with OpenCV and Node fixes
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0", "wget")
    # Added opencv-python-headless to the base pip install
    .pip_install("comfy-cli", "huggingface_hub", "hf_transfer", "opencv-python-headless")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}) 
    .run_commands(
        "comfy --skip-prompt install --nvidia",
        "rm -rf /root/comfy/ComfyUI/custom_nodes/*",
        
        # Clone Nodes
        "git clone https://github.com/ltdrdata/ComfyUI-Manager.git /root/comfy/ComfyUI/custom_nodes/ComfyUI-Manager",
        "git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /root/comfy/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite",
        "git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git /root/comfy/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper",
        
        # 2. FORCE install requirements for VHS and WanWrapper
        # This ensures cv2 and other missing libs are caught
        "pip install -r /root/comfy/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt",
        "pip install -r /root/comfy/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper/requirements.txt",
        
        # Ensure opencv-python-headless is used even if requirements.txt asks for the standard one
        "pip uninstall -y opencv-python opencv-contrib-python",
        "pip install opencv-python-headless",

        # Setup Auto-Load Workflow
        "mkdir -p /root/comfy/ComfyUI/user/default",
        "mkdir -p /root/comfy/ComfyUI/web/assets",
        "wget -O /tmp/rapid_workflow.json https://huggingface.co/Phr00t/WAN2.2-14B-Rapid-AllInOne/resolve/main/Mega-v3/Rapid-AIO-Mega.json",
        "cp /tmp/rapid_workflow.json /root/comfy/ComfyUI/user/default/default_graph.json",
        "cp /tmp/rapid_workflow.json /root/comfy/ComfyUI/web/assets/default_graph.json"
    )
)

app = modal.App(name="comfy-wan-rapid", image=image)
vol = modal.Volume.from_name("wan-rapid-volume", create_if_missing=True)

MODEL_FILENAME = "Mega-v12/wan2.2-rapid-mega-aio-nsfw-v12.2.safetensors"
LOCAL_FILENAME = "wan2.2-rapid-mega-aio-nsfw-v12.2.safetensors"

@app.function(volumes={"/data": vol}, timeout=1800)
def download_rapid_aio():
    from huggingface_hub import hf_hub_download
    checkpoint_path = "/data/models/checkpoints"
    os.makedirs(checkpoint_path, exist_ok=True)
    hf_hub_download(
        repo_id="Phr00t/WAN2.2-14B-Rapid-AllInOne",
        filename=MODEL_FILENAME,
        local_dir=checkpoint_path,
        resume_download=True
    )
    vol.commit()

@app.function(
    gpu="A100", 
    volumes={"/data": vol},
    timeout=3600,
    container_idle_timeout=300
)
@modal.web_server(8100, startup_timeout=600) 
def serve_comfy():
    checkpoints_dir = "/root/comfy/ComfyUI/models/checkpoints"
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    volume_file_path = f"/data/models/checkpoints/{MODEL_FILENAME}"
    symlink_path = os.path.join(checkpoints_dir, LOCAL_FILENAME)
    
    if not os.path.exists(symlink_path):
        os.symlink(volume_file_path, symlink_path)
    
    import subprocess
    # Verify cv2 is actually working before starting server
    try:
        import cv2
        print(f"✅ OpenCV (cv2) version {cv2.__version__} detected.")
    except ImportError:
        print("❌ CRITICAL: cv2 still not found. Check build logs.")

    print("🚀 Starting ComfyUI on port 8100...")
    subprocess.Popen(
        "python /root/comfy/ComfyUI/main.py --listen 0.0.0.0 --port 8100", 
        shell=True
    )