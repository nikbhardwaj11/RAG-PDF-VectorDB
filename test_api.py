import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.getenv("NVIDIA_BASE_URL"),
    api_key=os.getenv("NVIDIA_API_KEY")
)

chat_model = os.getenv("NVIDIA_CHAT_MODEL")
embed_model = os.getenv("NVIDIA_EMBED_MODEL")


# TEST 1: CHAT API
print("\nTesting Chat API...")

try:
    response = client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "user", "content": "What is RAG? Explain in one sentence."}
        ],
        max_tokens=100
    )

    print("CHAT API: WORKING")
    print("Response:", response.choices[0].message.content)

except Exception as e:
    print("CHAT API: FAILED")
    print("Error:", e)


# TEST 2: EMBEDDING API
print("\nTesting Embedding API...")

try:
    response = client.embeddings.create(
        model=embed_model,
        input=["Artificial Intelligence is transforming technology."],
        extra_body={"input_type": "passage"}
    )

    vector = response.data[0].embedding

    print("EMBEDDING API: WORKING")
    print("Vector dimensions:", len(vector))
    print("First 5 values:", vector[:5])

except Exception as e:
    print("EMBEDDING API: FAILED")
    print("Error:", e)