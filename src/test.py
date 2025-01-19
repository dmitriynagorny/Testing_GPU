import asyncio
import time
import json
from typing import List, Union, Dict
import openai

HOST = '1'
PORT = 8000
PROTOCOL = 'http'

BASE_URL = f"{PROTOCOL}://{HOST}:{PORT}/v1/"
MODEL = f"GameScribes/Mistral-Nemo-AWQ"
API_KEY = "<KEY>"


# Функция для выполнения одного асинхронного запроса к модели
async def make_request(client, model: str, prompt: str, max_tokens: int = 2048) -> Dict:
    start_time = time.time()
    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens
        )
        end_time = time.time()
        output_text = response.choices[0].message.content
        token_count = len(output_text.split())
        response_time = end_time - start_time
        return {
            "response": output_text,
            "token_count": token_count,
            "response_time": response_time,
        }
    except Exception as e:
        return {"error": str(e)}

# Основной метод для тестирования скорости генерации токенов
async def test_generation_speed(
    client, model: str, query: str, parallel_requests: int
) -> Dict[int, Dict]:
    results = {}

    # Ограничиваем число параллельных запросов с помощью семафора
    semaphore = asyncio.Semaphore(parallel_requests)

    async def sem_task(task):
        async with semaphore:
            return await task

    tasks = [
        sem_task(make_request(client, model, query))
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


# Пример использования
if __name__ == "__main__":

    test_query = "Привет!"
    parallel_requests = 5

    openai.api_key = API_KEY
    openai.api_base = BASE_URL
    client = openai

    # Запуск теста
    results = asyncio.run(test_generation_speed(client, MODEL, test_query, parallel_requests))

    with open('data.json', 'w') as f:
        json.dump(results, f)

    # Вывод результатов
    for idx, result in results.items():
        print(f"Request {idx}: {result}")
