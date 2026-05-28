from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

repo_id   = "divyabhangre/tourism-pkg-prediction-data"
repo_type = "dataset"

api = HfApi(token=os.getenv("HF_TOKEN"))

# ── Create repo if not exists ──
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"✅ Dataset repo already exists.")
except RepositoryNotFoundError:
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"✅ Dataset repo created.")

# ── Upload data folder ──
api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
print("✅ Data uploaded to HF dataset space!")
