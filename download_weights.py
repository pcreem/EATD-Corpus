import requests
import os
import hashlib

def verify_sha256(file_path, expected_sha256):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    file_hash = sha256_hash.hexdigest()
    return file_hash == expected_sha256

def download_model():
    url = "https://github.com/pcreem/EATD-Corpus/releases/download/v1.0.0/model_weights.pth"
    dest = "model_weights.pth"
    expected_sha256 = "02f3ffdd54b161379089ddfb318f3b231de4e6754a186962459c38178d305627"

    if not os.path.exists(dest):
        print("⏬ 正在下載模型權重...")
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(dest, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("✅ 模型權重下載完成。")

        print("🔍 驗證 SHA256...")
        if not verify_sha256(dest, expected_sha256):
            raise ValueError("❌ SHA256 驗證失敗！檔案可能已被破壞。")
        print("✅ SHA256 驗證成功。")
    else:
        print("✅ 權重已存在，跳過下載。")

if __name__ == "__main__":
    download_model()
