import google.generativeai as genai
import tkinter as tk
from tkinter import simpledialog, scrolledtext, messagebox
from cryptography.fernet import Fernet
import base64
import hashlib
import threading

# 解密函式 (來自 decrypt_key.py)
def get_key(password: str) -> bytes:
    return base64.urlsafe_b64encode(hashlib.sha256(password.encode()).digest())

def decrypt_key_file(encrypted_path: str, password: str) -> str:
    try:
        key = get_key(password)
        fernet = Fernet(key)
        with open(encrypted_path, "rb") as f:
            encrypted_data = f.read()
        decrypted = fernet.decrypt(encrypted_data)
        return decrypted.decode('utf-8').strip()
    except Exception as e:
        messagebox.showerror("錯誤", f"解密失敗：{e}")
        return None

# 聊天介面
class ChatApp:
    def __init__(self, master, model):
        self.model = model
        self.chat = model.start_chat()

        master.title("Gemini 聊天介面")
        self.text_area = scrolledtext.ScrolledText(master, wrap=tk.WORD, height=25, width=70, state=tk.DISABLED)
        self.text_area.pack(padx=10, pady=10)

        self.entry = tk.Entry(master, width=70)
        self.entry.pack(padx=10, pady=5)
        self.entry.bind("<Return>", self.send_message)

        self.sending = False  # 控制是否正在送訊息，避免重複送出

    def send_message(self, event=None):
        if self.sending:
            return  # 如果正在傳送訊息，忽略新請求
        user_input = self.entry.get().strip()
        if not user_input:
            return
        self.entry.delete(0, tk.END)
        self.append_text(f"你：{user_input}\n")
        self.sending = True

        # 建立執行緒執行送訊息動作
        thread = threading.Thread(target=self.get_response, args=(user_input,))
        thread.daemon = True
        thread.start()

    def get_response(self, user_input):
        try:
            response = self.chat.send_message(user_input)
            text = response.text.strip()
            self.append_text_threadsafe(f"Gemini：{text}\n\n")
        except Exception as e:
            self.append_text_threadsafe(f"[錯誤] 無法取得回應：{e}\n")
        finally:
            self.sending = False

    def append_text_threadsafe(self, text):
        # 非主執行緒要用 after() 呼叫主執行緒更新UI
        self.text_area.after(0, self.append_text, text)

    def append_text(self, text):
        self.text_area.config(state=tk.NORMAL)
        self.text_area.insert(tk.END, text)
        self.text_area.config(state=tk.DISABLED)
        self.text_area.see(tk.END)

# 主流程
def main():
    # 先隱藏主視窗，彈出密碼輸入框
    root = tk.Tk()
    root.withdraw()

    password = simpledialog.askstring("輸入密碼", "請輸入 key.txt.aes 解密密碼：", show="*")
    if not password:
        return

    api_key = decrypt_key_file("key.txt.aes", password)
    if not api_key:
        return

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("models/gemini-2.5-pro")

    # 顯示聊天視窗
    chat_window = tk.Tk()
    app = ChatApp(chat_window, model)
    chat_window.mainloop()

if __name__ == "__main__":
    main()
