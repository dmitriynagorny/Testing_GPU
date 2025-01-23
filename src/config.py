### input/response params

MESSAGES=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write 21 examples of sorting in python."},
    ]

MAX_TOKENS = 2048
# MAX_COMPLETION_TOKENS = 
TOTAL_REQUESTS = 5

### server params

HOST = '81.94.159.217'
PORT = 8000
PROTOCOL = 'http'

BASE_URL = f"{PROTOCOL}://{HOST}:{PORT}/v1/"
MODEL_NAME = f"GameScribes/Mistral-Nemo-AWQ"
API_KEY = "ollama"

### model params

TEMPERATURE = 0.3
TOP_P = 0.9