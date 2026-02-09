# from dotenv import load_dotenv
# from langchain.chat_models import init_chat_model

# load_dotenv()
# messages = [
#     {"role": "system", "content": "You are a helpful assistant that provides information based on search results."},
#     {"role": "user", "content": "What is the capital of France?"}
# ]
# llm = init_chat_model("meta/llama-3.1-8b-instruct", model_provider="nvidia")

# messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "What is the capital of France?"}
# ]

# reply = llm.invoke(messages)
# print("Final Answer:")
# print(reply.content)



    # return {"final_answer": final_answer, "messages": [{"role": "assistant", "content": final_answer}]}

from pymongo import MongoClient

# This will no longer give ECONNREFUSED
client = MongoClient("mongodb://localhost:27017/")

db = client["search_agent_db"]
collection = db["Data"]
collection.insert_one({"test": "Connection successful!"})
print("Connected successfully!")