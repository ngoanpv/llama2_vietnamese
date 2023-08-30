import requests

BASE_URL = "http://127.0.0.1:8686"

def test_chat_api():
    question = "Quốc khánh của Việt Nam diễn ra vào ngày nào?"
    response = requests.get(f"{BASE_URL}/chat/", params={"question": question})
    data = response.json()
    print("Chat API Response:")
    print(data["response"][0])
    print("\n" + "-"*50 + "\n")

def test_stream_chat_api():
    question = "Việt Nam có bao nhiêu tỉnh?"
    with requests.get(f"{BASE_URL}/stream_chat/", params={"question": question}, stream=True) as response:
        print("Stream Chat API Response:")
        buffer = ""
        for chunk in response.iter_content(chunk_size=1024):
            buffer += chunk.decode('utf-8')  
            while '\n' in buffer:
                line, _, buffer = buffer.partition('\n')
                print(line)
        if buffer: 
            print(buffer)
        print("\n" + "-"*50 + "\n")


if __name__ == "__main__":
    test_chat_api()
    test_stream_chat_api()
