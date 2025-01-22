import asyncio
import time
import json
from typing import List, Union, Dict

from openai import OpenAI



# Функция для выполнения одного Асинхронного запроса к модели
async def amake_request(client: OpenAI, model: str, messages: list, max_tokens: int = 2048) -> Dict:
    start_time = time.perf_counter()
    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=model,
            messages=messages,
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
    client, config
) -> Dict[int, Dict]:
    results = {}

    # Ограничиваем число параллельных запросов с помощью семафора
    semaphore = asyncio.Semaphore(config.TOTAL_REQUESTS)

    async def sem_task(task):
        async with semaphore:
            return await task

    tasks = [
        sem_task(amake_request(client, config.MODEL_NAME, config.MESSAGES))
        for _ in range(config.TOTAL_REQUESTS)
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
def make_request(client: OpenAI, model: str, messages: list, max_tokens: int = 2048) -> Dict:
    start_time = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
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
    client, config
) -> Dict[int, Dict]:
    
    results = {}

    # Сохраняем результаты
    for i in range(config.TOTAL_REQUESTS):
        response = make_request(client, config.MODEL_NAME, config.MESSAGES, config.MAX_TOKENS)
        try:
            results[i] = response
        except:
            results[i] = {"error": str(response)}

    return results



# Пример использования
if __name__ == "__main__":

    import config

    client = OpenAI(
        api_key=config.API_KEY,
        base_url=config.BASE_URL
        )

    # Запуск теста
    results = asyncio.run(atest_generation_speed(client, config))
    # results = test_generation_speed(client, config)

    # Вывод результатов
    for idx, result in results.items():
        print(f"Request {idx}: {result}")
