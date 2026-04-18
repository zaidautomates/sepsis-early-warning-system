import os
from huggingface_hub import HfFileSystem

def ensure_data_exists():
    """Checks for the clinical data matrix and pulls it from HF Storage if missing."""
    local_path = "data/X_seq.npy"
    
    # EXACT path to your file in the HF Bucket
    # Format: hf://buckets/OWNER/BUCKET_NAME/FILE_PATH
    remote_path = "hf://buckets/zaidautomates/sepsis-clinical-data/X_seq.npy"
    
    if not os.path.exists(local_path):
        print("--- [!] Data matrix missing. Fetching from Zaid Automates cloud... ---")
        
        # Ensure the local 'data' folder exists
        os.makedirs("data", exist_ok=True)
        
        try:
            fs = HfFileSystem()
            # This copies the file from the cloud to your local path
            fs.get(remote_path, local_path)
            print("--- [✓] Remote fetch complete. ---")
        except Exception as e:
            print(f"--- [X] Fetch failed: {e} ---")
            print("Verify that your bucket name and file path are exactly correct.")
    else:
        print("--- [i] Data matrix verified locally. ---")

if __name__ == "__main__":
    ensure_data_exists()