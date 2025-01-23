import asyncio
import time
from openai import AsyncOpenAI  # Используем асинхронный клиент
from typing import Dict, List

# Функция для выполнения запроса с измерением времени до первого токена
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

# Пример использования
if __name__ == "__main__":
    import config
    from openai import AsyncOpenAI  # Используем асинхронный клиент

    # Инициализация асинхронного клиента OpenAI
    client = AsyncOpenAI(
        api_key=config.E_API_KEY,
        base_url=config.E_BASE_URL
    )

    # Запуск теста
    try:
        result = asyncio.run(
            amake_request_with_ttft(
                client,
                model=config.E_MODEL_NAME,
                messages=config.E_MESSAGES,
                max_tokens=config.E_MAX_TOKENS,
                temperature=config.E_TEMPERATURE,
                top_p=config.E_TOP_P
            )
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        result = {}

    # Вывод результатов
    if "error" in result:
        print(f"Request failed with error: {result['error']}")
    else:
        print(f"Response: {result['response']}")
        print(f"Tokens generated: {result['token_count']}")
        print(f"Total response time: {result['response_time']:.2f} seconds")
        print(f"Time to first token: {result['time_to_first_token']:.2f} seconds")