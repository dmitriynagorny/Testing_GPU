import asyncio
import time
from typing import Dict

from openai import AsyncOpenAI
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.enums import ParseMode


import creds
import config


BASE_URL = f"{config.PROTOCOL}://{config.HOST}:{config.PORT}/v1/"

# Initialize the OpenAI client
client = AsyncOpenAI(api_key="v", base_url=BASE_URL)

# Initialize the bot and dispatcher
bot = Bot(token=creds.TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

# Function to make an asynchronous request to the OpenAI model
async def amake_request(client: AsyncOpenAI, model: str, messages: list, max_tokens: int = 2048, temperature: float = 0.3, top_p: float = 0.9) -> Dict:
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

# Command handler for the /start command
@dp.message(Command("start"))
async def send_welcome(message: Message):
    await message.reply("Hello! I'm your OpenAI-powered Telegram bot. Send me a message and I'll generate a response for you.")

# Message handler for processing user messages
@dp.message()
async def echo(message: Message):
    user_message = message.text

    # Prepare the messages for the OpenAI API
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_message}
    ]

    # Make the request to the OpenAI API
    response = await amake_request(client, model="/model", messages=messages)

    if "error" in response:
        await message.reply(f"Sorry, an error occurred: {response['error']}")
    else:
        await message.reply(response["response"], parse_mode=ParseMode.MARKDOWN)

# Start the bot
async def main():
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())