### input/response params

E_MESSAGES=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write 5 examples of sorting in python."},
    ]

E_MAX_TOKENS = 2048
# MAX_COMPLETION_TOKENS = 
E_TOTAL_REQUESTS = 2
E_REQUEST_ASYNC = True

### server params

E_HOST = 'localhost'
E_PORT = 8008
E_PROTOCOL = 'http'

E_BASE_URL = f"{E_PROTOCOL}://{E_HOST}:{E_PORT}/v1/"
E_MODEL_NAME = f"/model"
E_API_KEY = "ollama"

### model params

E_TEMPERATURE = 0.3
E_TOP_P = 0.9