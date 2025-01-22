### input/response params

MESSAGES=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The LA Dodgers won in 2020."},
        {"role": "user", "content": "Where was it played?"}
    ]

MAX_TOKENS = 2048
# MAX_COMPLETION_TOKENS = 
TOTAL_REQUESTS = 20

### server params

HOST = 'localhost'
PORT = 8000
PROTOCOL = 'http'

BASE_URL = f"{PROTOCOL}://{HOST}:{PORT}/v1/"
MODEL_NAME = f"GameScribes/Mistral-Nemo-AWQ"
API_KEY = "ollama"

### model params

TEMPERATURE = 0.3
TOP_P = 0.9