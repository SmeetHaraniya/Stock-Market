import os
import secrets

SPECIAL_KEY = os.getenv("SECRET_KEY", secrets.token_hex(32))
