import asyncio
import aiohttp
import time
import json
from tqdm import tqdm

# === НАСТРОЙКИ ===
MODE = "triton"  # "triton" или "app"
PATTERN = "mixed"  # "embeddings", "hybrid_score", "mixed"
CONCURRENCY = 10
TOTAL_REQUESTS = 1000

a = "что такое глубокое обучение?"
# === ДАННЫЕ ===
QUERIES = [a*100]
PASSAGES = [
    "нейросети имитируют работу мозга",
    "глубокое обучение — это часть машинного обучения",
    "глубокое обучение — это часть машинного обучения",
    "глубокое обучение — это часть машинного обучения",
    "глубокое обучение — это часть машинного обучения",
    "нейросети имитируют работу мозга"
]
TEXTS = [a*20]*10
WEIGHTS = [0.4, 0.2, 0.4]

# === ENDPOINTS ===
ENDPOINTS = {
    # "triton": "http://localhost:8999/v2/models/hybrid_model/infer",
    "triton": "https://ai-edu.mos.ru/ai-embedder-triton/v2/models/hybrid_model/infer",
    "app_embeddings": "http://localhost:8999/embeddings",
    "app_hybrid_score": "http://localhost:8999/hybrid_score"
}

def make_triton_payload(mode: str):
    if mode == "embeddings":
        return {
            "inputs": [
                {"name": "MODE", "datatype": "BYTES", "shape": [1, 1], "data": ["embeddings"]},
                {"name": "TEXTS", "datatype": "BYTES", "shape": [1, len(TEXTS)], "data": TEXTS},
                {"name": "TEXT", "datatype": "BYTES", "shape": [1, 1], "data": [""]},
                {"name": "QUERIES", "datatype": "BYTES", "shape": [1, 1], "data": [""]},
                {"name": "PASSAGES", "datatype": "BYTES", "shape": [1, 1], "data": [""]},
                {"name": "WEIGHTS", "datatype": "FP32", "shape": [1, 3], "data": [0.0, 0.0, 0.0]}
            ],
            "outputs": [{ "name": "EMBEDDINGS" }]
        }
    elif mode == "hybrid_score":
        return {
            "inputs": [
                {"name": "MODE", "datatype": "BYTES", "shape": [1, 1], "data": ["hybrid_score"]},
                {"name": "TEXTS", "datatype": "BYTES", "shape": [1, 1], "data": [""]},
                {"name": "TEXT", "datatype": "BYTES", "shape": [1, 1], "data": [""]},
                {"name": "QUERIES", "datatype": "BYTES", "shape": [1, len(QUERIES)], "data": QUERIES},
                {"name": "PASSAGES", "datatype": "BYTES", "shape": [1, len(PASSAGES)], "data": PASSAGES},
                {"name": "WEIGHTS", "datatype": "FP32", "shape": [1, 3], "data": WEIGHTS}
            ],
            "outputs": [{ "name": "SCORES" }]
        }

def make_app_payload(mode: str):
    if mode == "embeddings":
        return ENDPOINTS["app_embeddings"], { "texts": TEXTS }
    elif mode == "hybrid_score":
        return ENDPOINTS["app_hybrid_score"], {
            "queries": QUERIES,
            "passages": PASSAGES,
            "weights": WEIGHTS
        }

async def send_request(session, mode, result_dict):
    if MODE == "triton":
        url = ENDPOINTS["triton"]
        payload = make_triton_payload(mode)
    else:
        url, payload = make_app_payload(mode)

    async with session.post(url, json=payload) as resp:
        await resp.read()

async def worker(semaphore, session, mode, result_dict, pbar):
    async with semaphore:
        try:
            start = time.perf_counter()
            await send_request(session, mode, result_dict)
            duration = time.perf_counter() - start
            result_dict[mode].append(duration)
        except Exception as e:
            print(f"❌ Ошибка в worker ({mode}): {e}")
        finally:
            pbar.update(1)

async def run_benchmark():
    semaphore = asyncio.Semaphore(CONCURRENCY)
    result_times = { "embeddings": [], "hybrid_score": [] }

    async with aiohttp.ClientSession() as session:
        tasks = []
        with tqdm(total=TOTAL_REQUESTS, desc="Running benchmark") as pbar:
            for i in range(TOTAL_REQUESTS):
                if PATTERN == "mixed":
                    mode = "embeddings" if i % 2 == 0 else "hybrid_score"
                else:
                    mode = PATTERN

                tasks.append(worker(semaphore, session, mode, result_times, pbar))

            start = time.perf_counter()
            await asyncio.gather(*tasks)
            total = time.perf_counter() - start

    print("\n📊 Результаты:")
    print(f"Тип запуска: {MODE.upper()} | Pattern: {PATTERN}")
    print(f"Запросов всего: {TOTAL_REQUESTS}")
    print(f"Параллельность: {CONCURRENCY}")
    print(f"Общее время: {total:.2f} сек")
    print(f"RPS: {TOTAL_REQUESTS / total:.2f}\n")

    for mode in ["embeddings", "hybrid_score"]:
        times = result_times[mode]
        if times:
            avg = sum(times) / len(times)
            rps = len(times) / total
            print(f"--- {mode.upper()} ---")
            print(f"  Запросов: {len(times)}")
            print(f"  Среднее время: {avg:.4f} сек")
            print(f"  RPS: {rps:.2f}")
            print()

if __name__ == "__main__":
    asyncio.run(run_benchmark())
