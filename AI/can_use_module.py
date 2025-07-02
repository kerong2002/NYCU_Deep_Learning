import google.generativeai as genai

genai.configure(api_key="")

models = genai.list_models()

print("可用模型清單：")
for m in models:
    print(m.name)
