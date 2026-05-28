from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

repo_id   = "divyabhangre/tourism-package-predict"
repo_type = "space"

# ── Push all deployment files to HF Space ──
files_to_upload = [
    ("tourism_project/app.py",              "app.py"),
    ("tourism_project/Dockerfile",          "Dockerfile"),
    ("tourism_project/requirements.txt",    "requirements.txt"),
]

for local_path, repo_path in files_to_upload:
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=repo_path,
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print(f"✅ Uploaded {repo_path} to HF Space!")

print(f"\n🚀 All files pushed to HF Space!")
print(f"👉 Visit: https://huggingface.co/spaces/{repo_id}")
