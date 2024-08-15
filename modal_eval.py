# modal run --env hackathon --detach modal_eval

import modal
from pathlib import Path

VOL_NAME = "humaneval-x-evals"
VOL_MOUNT_PATH = Path("/vol")
MODEL_PATH = "/runs/model"
MINUTES = 60  # seconds
HOURS = 60 * MINUTES

MODEL_NAME = "openai/gpt-4o-mini-2024-07-18"
LANGUAGE = 'python'
APP_NAME = f"humaneval-x-{MODEL_NAME.split('/')[1]}"


eval_image = (
    modal.Image.from_registry(
        "rishubi/codegeex"
    )
    # modal.Image.from_registry(
    #     "ubuntu:22.04", add_python="3.11"
    # ).apt_install(
    #     "git",
    # )
    # .pip_install_private_repos(
    #     "github.com/devanshrj/llm-eval@humaneval-x",
    #     git_user="devanshrj",
    #     secrets=[modal.Secret.from_name("github-read-private")]
    # )
)

app = modal.App(name=APP_NAME, image=eval_image)
app = modal.App(
    APP_NAME,
    secrets=[
        modal.Secret.from_name("llm-provider-tokens"),
        modal.Secret.from_name("github-read-private"),
    ],
)

volume = modal.Volume.from_name(VOL_NAME, create_if_missing=True)

class Colors:
    """ANSI color codes"""

    GREEN = "\033[0;32m"
    BLUE = "\033[0;34m"
    GRAY = "\033[0;90m"
    BOLD = "\033[1m"
    END = "\033[0m"


@app.function(
    image=eval_image,
    timeout=24 * HOURS,
    volumes={VOL_MOUNT_PATH: volume},
)
def evaluate():
    import os
    from pathlib import Path
    import subprocess

    GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

    print("Evaluating model...")
    run_folder = Path("/vol")
    evals_folder = Path(run_folder) / "llm-eval"
    artifacts_folder = Path(run_folder) / "artifacts"

    if os.path.exists(evals_folder):
        RM_CMD = "rm -rf llm-eval"
        subprocess.call(RM_CMD.split(), cwd=run_folder)
        
    CLONE_CMD = f"git clone -b humaneval-x https://devanshrj:{GITHUB_TOKEN}@github.com/devanshrj/llm-eval.git"
    subprocess.call(CLONE_CMD.split(), cwd=run_folder)
    
    INSTALL_CMD = "pip install -r requirements.txt"
    subprocess.call(INSTALL_CMD.split(), cwd=evals_folder)

    EVAL_CMD = f"python main.py --model {MODEL_NAME} --language {LANGUAGE} --out-path {artifacts_folder}"
    subprocess.call(EVAL_CMD.split(), cwd=evals_folder)    
    volume.commit()

    print("evaluation complete!")