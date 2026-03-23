import modal
import os

# --- Configuration ---
# Volume for storing models and data persistently
volume = modal.Volume.from_name("comfy-wan-rapid-volume", create_if_missing=True)

# Hugging Face model details
MODEL_REPO_ID = "Phr00t/WAN2.2-14B-Rapid-AllInOne"
MODEL_FILENAME = "Mega-v12/wan2.2-rapid-mega-aio-nsfw-v12.2.safetensors"
MODEL_BASE_NAME = os.path.basename(MODEL_FILENAME)

# Paths within the container
COMFYUI_ROOT = "/root/comfy/ComfyUI"
CUSTOM_NODES_PATH = f"{COMFYUI_ROOT}/custom_nodes"
CHECKPOINTS_PATH = f"{COMFYUI_ROOT}/models/checkpoints"
VOLUME_DATA_PATH = "/data"
VOLUME_CHECKPOINTS_PATH = f"{VOLUME_DATA_PATH}/models/checkpoints"

# --- Custom Nodes ---
# Based on the nodes found in Rapid-AIO-Mega.json:
# - WanVideoVACEStartToEndFrame, WanVaceToVideo -> ComfyUI-WanVideoWrapper
# - VHS_VideoCombine -> ComfyUI-VideoHelperSuite
# - PrimitiveInt -> ComfyUI_Fill-Nodes (a common utility pack)
# - ComfyUI-Manager is included for convenience.
CUSTOM_NODE_REPOS = {
    "ComfyUI-Manager": "https://github.com/ltdrdata/ComfyUI-Manager.git",
    "ComfyUI-VideoHelperSuite": "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git",
    "ComfyUI-WanVideoWrapper": "https://github.com/kijai/ComfyUI-WanVideoWrapper.git",
    "ComfyUI_Fill-Nodes": "https://github.com/filliptm/ComfyUI_Fill-Nodes.git",
}

# --- Image Definition ---
# This defines the container environment for running ComfyUI.

# Create a list of commands to clone repositories
clone_cmds = [
    f"git clone {url} {CUSTOM_NODES_PATH}/{name}"
    for name, url in CUSTOM_NODE_REPOS.items()
]

# Create a list of commands to install Python requirements for the custom nodes
req_cmds = [
    f"pip install -r {CUSTOM_NODES_PATH}/ComfyUI-VideoHelperSuite/requirements.txt",
    f"pip install -r {CUSTOM_NODES_PATH}/ComfyUI-WanVideoWrapper/requirements.txt",
    f"pip install -r {CUSTOM_NODES_PATH}/ComfyUI_Fill-Nodes/requirements.txt",
]

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0", "wget")
    .pip_install(
        "comfy-cli",
        "huggingface_hub",
        "hf_transfer",
        "opencv-python-headless",
        "imageio-ffmpeg",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_commands(
        # Install ComfyUI using the comfy-cli tool
        "comfy --skip-prompt install --nvidia",
        # Ensure the custom_nodes directory is clean for idempotent git clones
        f"rm -rf {CUSTOM_NODES_PATH}/*",
        # Clone all custom node repositories in a single RUN command for layer efficiency
        " && ".join(clone_cmds),
        # Install requirements for all custom nodes
        *req_cmds,
    )
)

app = modal.App(name="comfy-wan-rapid", image=image)


@app.function(volumes={VOLUME_DATA_PATH: volume}, timeout=1800)
def download_model():
    """
    Downloads the model from Hugging Face Hub into the persistent volume.
    This function is run once to populate the volume.
    """
    from huggingface_hub import hf_hub_download

    # Ensure the target directory exists in the volume
    os.makedirs(VOLUME_CHECKPOINTS_PATH, exist_ok=True)

    print(f"Downloading {MODEL_FILENAME} to {VOLUME_CHECKPOINTS_PATH}...")
    hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=MODEL_FILENAME,
        local_dir=VOLUME_CHECKPOINTS_PATH,
        local_dir_use_symlinks=False,  # Download the file, don't symlink to cache
    )
    print("Download complete. Committing to volume.")
    volume.commit()


@app.function(
    gpu="A100",
    volumes={VOLUME_DATA_PATH: volume},
    timeout=3600,
    # Allow the container to stay up for 5 minutes after the last request
    # to keep the session warm for interactive use.
    container_idle_timeout=300,
)
@modal.web_server(8100, startup_timeout=600)
def serve_comfy():
    """
    Serves the ComfyUI web interface, with input/output and models
    mapped to the persistent volume.
    """
    import subprocess
    import shutil

    # --- Volume Mapping ---
    # Map ComfyUI's input and output directories to the persistent volume.
    # This ensures that files uploaded and generated are not lost.

    # 1. Define persistent paths on the volume and ensure they exist
    persistent_input_path = f"{VOLUME_DATA_PATH}/input"
    persistent_output_path = f"{VOLUME_DATA_PATH}/output"
    os.makedirs(persistent_input_path, exist_ok=True)
    os.makedirs(persistent_output_path, exist_ok=True)

    # 2. Define ComfyUI's local paths
    comfy_input_path = f"{COMFYUI_ROOT}/input"
    comfy_output_path = f"{COMFYUI_ROOT}/output"

    # 3. Symlink local paths to persistent volume paths for data persistence
    for local_path, persistent_path in [
        (comfy_input_path, persistent_input_path),
        (comfy_output_path, persistent_output_path),
    ]:
        if os.path.lexists(local_path):
            # Remove existing symlink or directory to avoid conflicts
            if os.path.islink(local_path):
                os.unlink(local_path)
            else:
                shutil.rmtree(local_path)
        os.symlink(persistent_path, local_path)
        print(f"Symlinked {local_path} -> {persistent_path}")

    # 4. Set permissions to allow the web server user to write to the volume
    subprocess.run(f"chmod -R 777 {VOLUME_DATA_PATH}", shell=True)

    # --- Model Symlinking ---
    # Symlink the downloaded model from the volume to ComfyUI's model directory.
    volume_model_path = f"{VOLUME_CHECKPOINTS_PATH}/{MODEL_FILENAME}"
    comfy_model_path = f"{CHECKPOINTS_PATH}/{MODEL_BASE_NAME}"

    if not os.path.lexists(comfy_model_path):
        os.symlink(volume_model_path, comfy_model_path)
        print(f"Symlinked model {comfy_model_path} -> {volume_model_path}")

    # --- Start ComfyUI ---
    print("🚀 Starting ComfyUI server on port 8100...")
    subprocess.Popen(
        f"python {COMFYUI_ROOT}/main.py --listen 0.0.0.0 --port 8100", shell=True
    )
