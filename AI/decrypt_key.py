from cryptography.fernet import Fernet
import base64
import hashlib
import os

# 將密碼轉換為 Fernet 金鑰
def get_key(password: str) -> bytes:
    return base64.urlsafe_b64encode(hashlib.sha256(password.encode()).digest())

# 解密 key.txt.aes
def decrypt_key_file(encrypted_path: str, password: str) -> str:
    try:
        key = get_key(password)
        fernet = Fernet(key)

        with open(encrypted_path, "rb") as f:
            encrypted_data = f.read()

        decrypted = fernet.decrypt(encrypted_data)
        return decrypted.decode('utf-8').strip()

    except Exception as e:
        return f"[錯誤] 解密失敗：{e}"

if __name__ == "__main__":
    aes_path = "key.txt.aes"
    if not os.path.exists(aes_path):
        print("[錯誤] 找不到 key.txt.aes 檔案")
    else:
        pwd = input("請輸入加密密碼：")
        result = decrypt_key_file(aes_path, pwd)
        print("\n===== 解密結果 =====")
        print(result)
