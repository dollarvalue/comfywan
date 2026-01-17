import modal
import os

# 1. Image definition with specific fixes for VideoHelperSuite
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0", "wget")
    # 1. Pre-install pip packages to avoid re-downloads
    .pip_install(
        "comfy-cli", "huggingface_hub", "hf_transfer", 
        "opencv-python-headless", "imageio-ffmpeg"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}) 
    .run_commands(
        # 2. Use --yes and --skip-prompt to prevent interactive hang-ups
        "comfy --skip-prompt install --nvidia",
        
        # 3. Clone all nodes in ONE command to optimize layer caching
        "cd /root/comfy/ComfyUI/custom_nodes && "
        "git clone https://github.com/ltdrdata/ComfyUI-Manager.git && "
        "git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && "
        "git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git",
        
        # 4. Install all requirements at once
        "pip install -r /root/comfy/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt",
        "pip install -r /root/comfy/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper/requirements.txt"
    )
)

app = modal.App(name="comfy-wan-rapid", image=image)
vol = modal.Volume.from_name("wan-rapid-volume", create_if_missing=True)

MODEL_FILENAME = "Mega-v12/wan2.2-rapid-mega-aio-nsfw-v12.2.safetensors"

@app.function(volumes={"/data": vol}, timeout=1800)
def download_rapid_aio():
    from huggingface_hub import hf_hub_download
    checkpoint_path = "/data/models/checkpoints"
    os.makedirs(checkpoint_path, exist_ok=True)
    hf_hub_download(
        repo_id="Phr00t/WAN2.2-14B-Rapid-AllInOne",
        filename=MODEL_FILENAME,
        local_dir=checkpoint_path
        # resume_download removed to fix DeprecationWarning
    )
    vol.commit()

@app.function(
    gpu="A100", 
    volumes={"/data": vol},
    timeout=3600,
    scaledown_window=300
)
@modal.web_server(8100, startup_timeout=600) 
def serve_comfy():
    import subprocess
    import os
    import shutil

    # 1. Setup the persistent input directory on the Volume
    persistent_input = "/data/input"
    os.makedirs(persistent_input, exist_ok=True)

    # 1. Setup Persistent Output Directory
    persistent_output = "/data/output"
    os.makedirs(persistent_output, exist_ok=True)

    # 2. Fix permissions: Modal volumes need explicit permission checks 
    # for the web-server user to write files via the UI.
    subprocess.run(f"chmod -R 777 {persistent_input}", shell=True)

    # 3. Direct ComfyUI to use the Volume folder as its 'input' source
    # We remove the local 'input' folder and symlink the WHOLE directory
    comfy_input_path = "/root/comfy/ComfyUI/input"
    if os.path.islink(comfy_input_path) or os.path.exists(comfy_input_path):
        import shutil
        if os.path.islink(comfy_input_path):
            os.unlink(comfy_input_path)
        else:
            shutil.rmtree(comfy_input_path)

    # 2. Map ComfyUI 'output' to the Volume
    comfy_output_path = "/root/comfy/ComfyUI/output"
    if os.path.lexists(comfy_output_path):
        if os.path.islink(comfy_output_path):
            os.unlink(comfy_output_path)
        else:
            shutil.rmtree(comfy_output_path)
    
    os.symlink(persistent_output, comfy_output_path)

    # This makes the Volume folder THE input folder
    os.symlink(persistent_input, comfy_input_path)

    # 4. Handle Model Checkpoint Symlink
    checkpoints_dir = "/root/comfy/ComfyUI/models/checkpoints"
    os.makedirs(checkpoints_dir, exist_ok=True)
    volume_file_path = f"/data/models/checkpoints/{MODEL_FILENAME}"
    symlink_path = os.path.join(checkpoints_dir, "wan2.2-rapid-mega-aio-nsfw-v12.2.safetensors")
    
    if not os.path.lexists(symlink_path):
        os.symlink(volume_file_path, symlink_path)

    print("🚀 Starting ComfyUI on port 8100...")
    subprocess.Popen(
        "python /root/comfy/ComfyUI/main.py --listen 0.0.0.0 --port 8100", 
        shell=True
    )