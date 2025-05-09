# login.py

import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

# Load config
with open("config.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

# Create authenticator object
authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"]
)

# Export authenticator object
