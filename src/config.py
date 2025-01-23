### input/response params

MESSAGES=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write 21 examples of sorting in python."},
    ]

MAX_TOKENS = 2048
# MAX_COMPLETION_TOKENS = 
TOTAL_REQUESTS = 2
REQUEST_ASYNC = True

### server params

HOST = 'localhost'
PORT = 8008
PROTOCOL = 'http'

BASE_URL = f"{PROTOCOL}://{HOST}:{PORT}/v1/"
MODEL_NAME = f"/model"
API_KEY = "ollama"

### model params

TEMPERATURE = 0.3
TOP_P = 0.9