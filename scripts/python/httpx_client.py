import httpx

def chat_completion(url: str, payload: dict):
    with httpx.Client() as client:
        response = client.post(url, json=payload)
        if response.status_code == 200:
            print(response.text)
        else:
            print(f"Error: {response.status_code}")
            print(response.text)

# Example usage
if __name__ == "__main__":
    url = "http://localhost:6979/v1/chat/completions"
    payload = {
        "messages": [{"role": "user", "content": "Hello!"}],
        "model": "phi3-mini-int4",
        "max_tokens": 80,
        "temperature": 0.0,
        "stream": False  # Set stream to False
    }
    chat_completion(url, payload)