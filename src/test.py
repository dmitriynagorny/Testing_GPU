import asyncio
import time
import json
from typing import List, Union, Dict
from openai import OpenAI

HOST = 'localhost'
PORT = 8000
PROTOCOL = 'http'

BASE_URL = f"{PROTOCOL}://{HOST}:{PORT}/v1/"
MODEL = f"GameScribes/Mistral-Nemo-AWQ"
API_KEY = "ollama"


# Функция для выполнения одного Асинхронного запроса к модели
async def amake_request(client: OpenAI, model: str, prompt: str, max_tokens: int = 4096) -> Dict:
    start_time = time.perf_counter()
    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Who won the world series in 2020?"},
                {"role": "assistant", "content": "The LA Dodgers won in 2020."},
                {"role": "user", "content": "Where was it played?"}
            ],
            max_tokens=max_tokens
        )
        end_time = time.perf_counter()

        output_text = response.choices[0].message.content
        token_count = response.usage.completion_tokens
        response_time = end_time - start_time
        
        return {
            "response": output_text,
            "token_count": token_count,
            "response_time": response_time,
        }
    except Exception as e:
        return {"error": str(e)}

# Основной метод для тестирования скорости генерации токенов
async def atest_generation_speed(
    client, model: str, query: str, parallel_requests: int
) -> Dict[int, Dict]:
    results = {}

    # Ограничиваем число параллельных запросов с помощью семафора
    semaphore = asyncio.Semaphore(parallel_requests)

    async def sem_task(task):
        async with semaphore:
            return await task

    tasks = [
        sem_task(amake_request(client, model, query))
        for _ in range(parallel_requests)
    ]

    responses = await asyncio.gather(*tasks, return_exceptions=True)

    # Сохраняем результаты
    for idx, response in enumerate(responses):
        if isinstance(response, Exception):
            results[idx] = {"error": str(response)}
        else:
            results[idx] = response

    return results


# CbpФункция для выполнения одного Синхронного запроса к модели
def make_request(client: OpenAI, model: str, prompt: str, max_tokens: int = 2048) -> Dict:
    start_time = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Who won the world series in 2020?"},
                {"role": "assistant", "content": "The LA Dodgers won in 2020."},
                {"role": "user", "content": "Where was it played?"}
            ],
            max_tokens=max_tokens
        )
        end_time = time.perf_counter()

        output_text = response.choices[0].message.content
        token_count = response.usage.completion_tokens
        response_time = end_time - start_time
        
        return {
            "response": output_text,
            "token_count": token_count,
            "response_time": response_time,
        }
    except Exception as e:
        return {"error": str(e)}
    

def test_generation_speed(
    client, model: str, query: str, parallel_requests: int
) -> Dict[int, Dict]:
    
    results = {}

    # Сохраняем результаты
    for i in range(parallel_requests):
        response = make_request(client, model, query)
        try:
            results[i] = response
        except:
            results[i] = {"error": str(response)}

    return results



# Пример использования
if __name__ == "__main__":

    test_query = "Привет!"
    parallel_requests = 5

    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
        )

    # Запуск теста
    # results = asyncio.run(atest_generation_speed(client, MODEL, test_query, parallel_requests))
    results = test_generation_speed(client, MODEL, test_query, parallel_requests)

    # with open('data.json', 'w') as f:
    #     json.dump(results, f)

    # Вывод результатов
    for idx, result in results.items():
        print(f"Request {idx}: {result}")
