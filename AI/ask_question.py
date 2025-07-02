import google.generativeai as genai
import os

api_key = ''
genai.configure(api_key = api_key)

model = genai.GenerativeModel('models/gemini-2.5-pro')
response = model.generate_content('你可以跟我問答嗎')

print(response.text)