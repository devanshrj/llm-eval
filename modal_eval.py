# modal run --env hackathon --detach modal_eval

import modal
from pathlib import Path

VOL_NAME = "humanevalx-final"
VOL_MOUNT_PATH = Path("/vol")
MODEL_PATH = "/runs/model"
MINUTES = 60  # seconds
HOURS = 60 * MINUTES

LANGUAGE = 'rust'
PROVIDER = 'mistral'

if PROVIDER == 'openai':
    MODELS = [
        'openai/gpt-4o-2024-05-13',
        'openai/gpt-4-turbo-2024-04-09',
        'openai/gpt-4o-mini-2024-07-18',
        'openai/gpt-4o-2024-08-06'
    ]
    APP_NAME = f"humanevalx-openai-{LANGUAGE}"
elif PROVIDER == 'anthropic':
    MODELS = [
        'anthropic/claude-3-5-sonnet-20240620',
        'anthropic/claude-3-opus-20240229',
    ]
    APP_NAME = f"humanevalx-anthropic-{LANGUAGE}"
elif PROVIDER == 'llama':
    MODELS = [
        'fireworksai/llama-3.1-70b-instruct',
        'fireworksai/llama-3.1-405b-instruct',
    ]
    APP_NAME = f"humanevalx-llama-{LANGUAGE}"
elif PROVIDER == 'mistral':
    MODELS = ['mistral/mistral-large-2407']
    APP_NAME = f"humanevalx-mistral-{LANGUAGE}"

eval_image = (
    modal.Image.from_registry(
        "rishubi/codegeex"
    )
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
    exec_folder = Path(run_folder) / "executions"
    artifacts_folder = Path(run_folder) / "artifacts"

    if os.path.exists(evals_folder):
        RM_CMD = "rm -rf llm-eval"
        subprocess.call(RM_CMD.split(), cwd=run_folder)
    
    if not os.path.exists(evals_folder):
        CLONE_CMD = f"git clone -b humaneval-x https://devanshrj:{GITHUB_TOKEN}@github.com/devanshrj/llm-eval.git"
        subprocess.call(CLONE_CMD.split(), cwd=run_folder)
    
    INSTALL_CMD = "pip install -r requirements.txt"
    subprocess.call(INSTALL_CMD.split(), cwd=evals_folder)

    for MODEL_NAME in MODELS:
        EVAL_CMD = f"python main.py --model {MODEL_NAME} --language {LANGUAGE} --out-path {artifacts_folder} --tmp-path {exec_folder}"
        subprocess.call(EVAL_CMD.split(), cwd=evals_folder)    
        volume.commit()

    print("evaluation complete!")