import httpx
import asyncio

async def fetch_models():
    url = "http://localhost:6979/v1/models"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()

async def main():
    models = await fetch_models()
    print(models)

if __name__ == "__main__":
    asyncio.run(main())