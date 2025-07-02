from cryptography.fernet import Fernet
import base64
import hashlib

# 產生 key：由密碼轉換成 32-byte key
def get_key(password: str) -> bytes:
    return base64.urlsafe_b64encode(hashlib.sha256(password.encode()).digest())

# 加密 key.txt 並寫入 key.txt.aes
def encrypt_file(password: str, input_file="key.txt", output_file="key.txt.aes"):
    key = get_key(password)
    fernet = Fernet(key)

    with open(input_file, "rb") as f:
        data = f.read()

    encrypted = fernet.encrypt(data)

    with open(output_file, "wb") as f:
        f.write(encrypted)

    print(f"✅ 已加密並儲存為：{output_file}")

# 測試用
if __name__ == "__main__":
    pwd = input("請輸入加密密碼：")
    encrypt_file(pwd)
