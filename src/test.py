import os
import asyncio
import time
import json
from datetime import datetime
from typing import List, Union, Dict

from openai import OpenAI
from openai import AsyncOpenAI  # Используем асинхронный клиент



async def amake_request_with_ttft(
    client: AsyncOpenAI,  # Используем асинхронный клиент
    model: str, 
    messages: List[Dict], 
    max_tokens: int = 2048, 
    temperature: float = 0.3, 
    top_p: float = 0.9
) -> Dict:
    start_time = time.perf_counter()  # Фиксируем время начала запроса
    first_token_time = None  # Время получения первого токена
    output_text = ""  # Собранный текст ответа
    token_count = 0  # Счетчик токенов

    try:
        # Используем stream=True для получения ответа по частям
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True  # Включаем потоковый режим
        )

        # Обрабатываем потоковые данные с помощью async for
        async for chunk in response:
            if not first_token_time:
                first_token_time = time.perf_counter()  # Фиксируем время первого токена
            
            if chunk.choices and chunk.choices[0].delta.content:
                output_text += chunk.choices[0].delta.content
                token_count += 1

        end_time = time.perf_counter()  # Фиксируем время завершения запроса

        return {
            "response": output_text,
            "token_count": token_count,
            "response_time": end_time - start_time,  # Общее время выполнения
            "time_to_first_token": first_token_time - start_time if first_token_time else None,  # Время до первого токена
        }
    except Exception as e:
        return {"error": str(e)}
    

async def amake_request(
    client: AsyncOpenAI,
    model: str,
    messages: list,
    max_tokens: int = 2048,
    temperature: float = 0.3,
    top_p: float = 0.9
) -> Dict:
    start_time = time.perf_counter()
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        end_time = time.perf_counter()

        return {
            "response": response.choices[0].message.content,
            "token_count": response.usage.completion_tokens,
            "response_time": end_time - start_time,
        }
    except Exception as e:
        return {"error": str(e)}


async def atest_generation_speed(
    client: AsyncOpenAI,
    config: object
) -> Dict[int, Dict]:
    # Параметры запроса
    request_params = {
        "model": config.E_MODEL_NAME,
        "messages": config.E_MESSAGES,
        "max_tokens": config.E_MAX_TOKENS,
        "temperature": config.E_TEMPERATURE,
        "top_p": config.E_TOP_P
    }

    # Создаем задачи для всех запросов
    tasks = [
        amake_request(client, **request_params)
        for _ in range(config.E_TOTAL_REQUESTS)
    ]

    # Выполняем все задачи одновременно
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    # Формируем результат
    return {
        idx: response if not isinstance(response, Exception)
        else {"error": str(response)}
        for idx, response in enumerate(responses)
    }


# Функция для выполнения одного Асинхронного запроса к модели
async def amake_sem_request(client: AsyncOpenAI, model: str, messages: list, max_tokens: int = 2048, temperature: float = 0.3, top_p: float = 0.9) -> Dict:
    start_time = time.perf_counter()
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
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
async def atest_sem_generation_speed(
    client, config
) -> Dict[int, Dict]:
    
    result = {}

    # Ограничиваем число параллельных запросов с помощью семафора
    semaphore = asyncio.Semaphore(config.E_TOTAL_REQUESTS)

    async def sem_task(task):
        async with semaphore:
            return await task

    tasks = [
        sem_task(
            amake_sem_request(
                client, 
                config.E_MODEL_NAME, config.E_MESSAGES, config.E_MAX_TOKENS, config.E_TEMPERATURE, config.E_TOP_P
                )
            )
        for _ in range(config.E_TOTAL_REQUESTS)
    ]

    responses = await asyncio.gather(*tasks, return_exceptions=True)

    # Сохраняем результаты
    for idx, response in enumerate(responses):
        if isinstance(response, Exception):
            result[idx] = {"error": str(response)}
        else:
            result[idx] = response

    return result



# CbpФункция для выполнения одного Синхронного запроса к модели
def make_request(client: OpenAI, model: str, messages: list, max_tokens: int = 2048, temperature: float = 0.3, top_p: float = 0.9) -> Dict:
    start_time = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
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
    
    result = {}

    # Сохраняем результаты
    for i in range(config.E_TOTAL_REQUESTS):
        response = make_request(
            client, 
            config.E_MODEL_NAME, config.E_MESSAGES, config.E_MAX_TOKENS, config.E_TEMPERATURE, config.E_TOP_P
            )
        try:
            result[i] = response
        except:
            result[i] = {"error": str(response)}

    return result


def save_with_timestamp(results, folder='results'):
    """
    Save the given results to a JSON file with a timestamp in the specified folder.

    Args:
        results (dict): The data to save.
        folder (str): The folder where the file will be saved. Defaults to 'results'.
    """
    # Create the folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Generate a timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Create the filename with the timestamp
    filename = f'data_{timestamp}.json'

    # Specify the full path to save the file
    file_path = os.path.join(folder, filename)

    # Write the data to the file
    with open(file_path, 'w') as f:
        json.dump(results, f)

    print(f'File saved: {file_path}')



# Пример использования
if __name__ == "__main__":

    import config

    config_envs = {i:vars(config)[i] for i in vars(config) if 'E_' in i}
    results = {"config":config_envs}

    client = AsyncOpenAI(
        api_key=config.E_API_KEY,
        base_url=config.E_BASE_URL
        )

    if config.E_REQUEST_ASYNC:
        results['main_test'] = asyncio.run(atest_generation_speed(client, config))
    elif config.E_SEMAPHORE_TEST:
        results['main_test'] = asyncio.run(atest_sem_generation_speed(client, config))
    else:
        results['main_test'] = test_generation_speed(client, config)

    # время до первого токена
    result_ttft = asyncio.run(
        amake_request_with_ttft(
            client,
            model=config.E_MODEL_NAME,
            messages=config.E_MESSAGES,
            max_tokens=config.E_MAX_TOKENS,
            temperature=config.E_TEMPERATURE,
            top_p=config.E_TOP_P
        )
    )

    results["ttft_test"] = result_ttft

    save_with_timestamp(results, folder='results')  # Saves to a custom folder

    # Вывод результатов
    for idx, result in results.items():
        print(f"Request {idx}: {result}")
