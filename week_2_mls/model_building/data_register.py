from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os


repo_id = "wankhedes27/machine-failure-prediction"
repo_type = "dataset"

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{wankhedes27}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{wankhedes27}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{wankhedes27}' created.")

api.upload_folder(
    folder_path="week_2_mls/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
