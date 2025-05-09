import google.generativeai as genai

# Configure API key
genai.configure(api_key="AIzaSyBW2WxRcKSh8aC06hFsJ0U5mpP0IUkONr8")

# List available models
models = genai.list_models()

# Print available models
for model in models:
    print(model.name)
