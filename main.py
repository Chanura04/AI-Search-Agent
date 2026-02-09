from dotenv import load_dotenv
from typing import Annotated,List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

load_dotenv()

llm = init_chat_model("deepseek-ai/deepseek-v3.2")

class State(TypedDict):
    messages: Annotated[list, add_messages]        #this is the message usually send into the graph to process and getting information from coming up with an answer to the user query
    user_qestion:str | None = None                      #this is the user query that we want to answer, it is optional because we can also use the messages to get the user query
    google_results: list | None = None                     #this is the google search results that we can use to answer the user query, it is optional because we can also use the messages to get the google search results
    bing_results: list | None = None                      #this is the bing search results that we can use to answer the user query, it is optional because we can also use the messages to get the bing search results
    reddit_results: list | None = None                    #this is the reddit search results that we can use to answer the user query, it is optional because we can also use the messages to get the reddit search results
    selected_reddut_urls: list[str] | None = None          #this is the selected reddit urls that we can use to answer the user query, it is optional because we can also use the messages to get the selected reddit urls
    reddit_post_data: list | None = None                      #this is the reddit post data that we can use to answer the user query, it is optional because we can also use the messages to get the reddit post data
    google_analysis: str | None = None                      #this is the google search results analysis that we can use to answer the user query, it is optional because we can also use the messages to get the google search results analysis
    bing_analysis: str | None = None                       #this is the bing search results analysis that we can use to answer the user query, it is optional because we can also use the messages to
    reddit_analysis: str | None = None                    #this is the reddit search results analysis that we can use to answer the user query, it is optional because we can also use the messages to get the reddit search results analysis   
    final_answer: str | None = None                      #this is the final answer that we can use to answer the user query, it is optional because we can also use the messages to get the final answer


def google_search(state:State):
    user_question = state.get("user_qestion","")
    print(f"Performing Google search for: {user_question}")
    google_result=[]
    return {"google_results": google_result}

def bing_search(state:State):
    user_question = state.get("user_qestion","")
    print(f"Performing Bing search for: {user_question}")
    bing_result=[]
    return {"bing_results": bing_result} 

def reddit_search(state:State):
    user_question = state.get("user_qestion","")
    print(f"Performing Reddit search for: {user_question}")
    reddit_result=[]
    return {"reddit_results": reddit_result}


def analyse_reddit_posts(state:State):
    return {"selected_reddut_urls":[]   }

def retreive_reddit_post_data(state:State):
    return {"reddit_post_data":[]}


def analyse_google_results(state:State):
    return {"google_analysis":""}

def analyse_bing_results(state:State):
    return {"bing_analysis":""}

def analyse_reddit_results(state:State):
    return {"reddit_analysis":""}

def synthesize_final_answer(state:State):
    return {"final_answer":""}



graph_builder= StateGraph(State)

graph_builder.add_node("google_search", google_search)
graph_builder.add_node("bing_search", bing_search)
graph_builder.add_node("reddit_search", reddit_search)
graph_builder.add_node("analyse_reddit_posts", analyse_reddit_posts)
graph_builder.add_node("retrieve_reddit_post_data", retreive_reddit_post_data)
graph_builder.add_node("analyse_google_results", analyse_google_results)
graph_builder.add_node("analyse_bing_results", analyse_bing_results)
graph_builder.add_node("analyse_reddit_results", analyse_reddit_results)
graph_builder.add_node("synthesize_final_answer", synthesize_final_answer)


'''connect nodes''' 

#these three will hapen is theat exact same time will execute all three operations parrelly and then move to the next step which is analysing the results
graph_builder.add_edge(START, "google_search")
graph_builder.add_edge(START, "bing_search")
graph_builder.add_edge(START, "reddit_search")

#we connect with reddit post analysis and data retrieval because we want to get the reddit post data and analyse it before we move to the final answer synthesis step
#until reddit post ready others are wating
graph_builder.add_edge("google_search", "analyse_reddit_posts")
graph_builder.add_edge("bing_search", "analyse_reddit_posts")
graph_builder.add_edge("reddit_search", "analyse_reddit_posts")
graph_builder.add_edge("analyse_reddit_posts", "retrieve_reddit_post_data")


graph_builder.add_edge("retrieve_reddit_post_data", "analyse_google_results")
graph_builder.add_edge("retrieve_reddit_post_data", "analyse_bing_results")
graph_builder.add_edge("retrieve_reddit_post_data", "analyse_reddit_results")

graph_builder.add_edge("analyse_google_results", "synthesize_final_answer")
graph_builder.add_edge("analyse_bing_results", "synthesize_final_answer")
graph_builder.add_edge("analyse_reddit_results", "synthesize_final_answer")

graph_builder.add_edge("synthesize_final_answer", END)


graph = graph_builder.compile()

def run_chatbot():
    print("Welcome to the AI Search Agent! Please enter your query:")
    
    while True:
        user_query = input("Ask a question (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            print("Goodbye!")
            break

        state={
            "messages":[{"role":"user","content":user_query}],
            "user_qestion":user_query,
            "google_results":None,
            "bing_results":None,
            "reddit_results":None,
            "google_analysis":None,
            "bing_analysis":None,
            "selected_reddut_urls":None,
            "reddit_post_data":None,
            "reddit_analysis":None,
            "final_answer":None
            
        }

        print("\n Starting parelle reasearch process...")
        print("Launchiung google search, bing search and reddit search...")
        final_state = graph.invoke(state)

        if final_state.get("final_answer"):
            print("\nFinal Answer:")
            print(f"\n Final Answer: \n {final_state.get('final_answer')}\n")
        print("-"*80)


if __name__ == "__main__":
    run_chatbot()