from dotenv import load_dotenv
from typing import Annotated,List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from web_operations import serp_search,reddit_search_api,reddit_post_retrieval
from prompts import PromptTemplates, get_google_analysis_messages, get_bing_analysis_messages, get_reddit_analysis_messages, get_synthesis_messages,get_reddit_url_analysis_messages

load_dotenv()

llm = init_chat_model("meta/llama-3.1-8b-instruct", model_provider="nvidia")

'''agent's working memory'''
class State(TypedDict):
    messages: Annotated[list, add_messages]                 # Conversation history
    user_qestion:str | None = None                          # Current user query
    google_results: list | None = None                      # Raw Google search results
    bing_results: list | None = None                        # Raw Bing search results
    reddit_results: list | None = None                      # Raw Reddit search results
    selected_reddut_urls: list[str] | None = None           # URLs chosen for deeper analysis
    reddit_post_data: list | None = None                    # Detailed Reddit post content
    google_analysis: str | None = None                      # Analyzed Google insights
    bing_analysis: str | None = None                        # Analyzed Bing insights
    reddit_analysis: str | None = None                      # Analyzed Reddit insights
    final_answer: str | None = None                         # Final synthesized answer to user question



'''structured output for reddit url analysis'''
class RedditURLAnalysis(BaseModel):
    selected_urls: List[str] = Field(description="List of Reddit URLs that contain valuable information for answering the user's question")


def google_search(state:State):
    user_question = state.get("user_qestion","")
    print(f"Performing Google search for: {user_question}")
    google_result=serp_search(user_question,engine="google")
    print(f"\n\nGoogle search results: {google_result}")
    return {"google_results": google_result}

def bing_search(state:State):
    user_question = state.get("user_qestion","")
    print(f"Performing Bing search for: {user_question}")
    bing_result= serp_search(user_question,engine="bing")
    print(f"\n\nBing search results: {bing_result}")
    return {"bing_results": bing_result} 

def reddit_search(state:State):
    user_question = state.get("user_qestion","")
    print(f"Performing Reddit search for: {user_question}")
    reddit_result=reddit_search_api(user_question)
    print(f"\n\nReddit search results: {reddit_result}")
    return {"reddit_results": reddit_result}


def analyze_reddit_posts(state:State):
    user_question = state.get("user_question", "")
    reddit_results = state.get("reddit_results", "")

    if not reddit_results:
        return {"selected_reddit_urls": []}
    

    '''get structured output for reddit url analysis'''
    structured_llm = llm.with_structured_output(RedditURLAnalysis)
    messages = get_reddit_url_analysis_messages(user_question, reddit_results)

    try:
        analysis = structured_llm.invoke(messages)
        selected_urls = analysis.selected_urls

        print("Selected URLs:")
        for i, url in enumerate(selected_urls, 1):
            print(f"   {i}. {url}")

    except Exception as e:
        print(e)
        selected_urls = []

    return {"selected_reddit_urls": selected_urls}


'''
    Fetches full post content and comments from the selected Reddit URLs
                                                                            '''

def retrieve_reddit_post_data(state:State):
    print("Getting reddit post comments")

    selected_urls = state.get("selected_reddit_urls", [])

    if not selected_urls:
        return {"reddit_post_data": []}

    print(f"Processing {len(selected_urls)} Reddit URLs")

    reddit_post_data = reddit_post_retrieval(selected_urls)

    if reddit_post_data:
        print(f"Successfully got {len(reddit_post_data)} posts")
    else:
        print("Failed to get post data")
        reddit_post_data = []

    print(reddit_post_data)
    return {"reddit_post_data": reddit_post_data}


def analyze_google_results(state: State):
    print("Analyzing google search results")

    user_question = state.get("user_question", "")
    google_results = state.get("google_results", "")

    messages = get_google_analysis_messages(user_question, google_results)
    reply = llm.invoke(messages)

    return {"google_analysis": reply.content}


def analyze_bing_results(state: State):
    print("Analyzing bing search results")

    user_question = state.get("user_question", "")
    bing_results = state.get("bing_results", "")

    messages = get_bing_analysis_messages(user_question, bing_results)
    reply = llm.invoke(messages)

    return {"bing_analysis": reply.content}


def analyze_reddit_results(state: State):
    print("Analyzing reddit search results")

    user_question = state.get("user_question", "")
    reddit_results = state.get("reddit_results", "")
    reddit_post_data = state.get("reddit_post_data", "")

    messages = get_reddit_analysis_messages(user_question, reddit_results, reddit_post_data)
    reply = llm.invoke(messages)

    return {"reddit_analysis": reply.content}


def synthesize_analyses(state: State):
    print("Combine all results together")

    user_question = state.get("user_question", "")
    google_analysis = state.get("google_analysis", "")
    bing_analysis = state.get("bing_analysis", "")
    reddit_analysis = state.get("reddit_analysis", "")

    messages = get_synthesis_messages(
        user_question, google_analysis, bing_analysis, reddit_analysis
    )

    reply = llm.invoke(messages)
    final_answer = reply.content

    return {"final_answer": final_answer, "messages": [{"role": "assistant", "content": final_answer}]}




graph_builder= StateGraph(State)

graph_builder.add_node("google_search", google_search)
graph_builder.add_node("bing_search", bing_search)
graph_builder.add_node("reddit_search", reddit_search)
graph_builder.add_node("analyze_reddit_posts", analyze_reddit_posts)
graph_builder.add_node("retrieve_reddit_post_data", retrieve_reddit_post_data)
graph_builder.add_node("analyze_google_results", analyze_google_results)
graph_builder.add_node("analyze_bing_results", analyze_bing_results)
graph_builder.add_node("analyze_reddit_results", analyze_reddit_results)
graph_builder.add_node("synthesize_final_answer", synthesize_analyses)


'''connect nodes''' 

#these three will hapen is theat exact same time will execute all three operations parrelly and then move to the next step which is analysing the results
graph_builder.add_edge(START, "google_search")
graph_builder.add_edge(START, "bing_search")
graph_builder.add_edge(START, "reddit_search")

#we connect with reddit post analysis and data retrieval because we want to get the reddit post data and analyse it before we move to the final answer synthesis step
#until reddit post ready others are wating
graph_builder.add_edge("google_search", "analyze_reddit_posts")
graph_builder.add_edge("bing_search", "analyze_reddit_posts")
graph_builder.add_edge("reddit_search", "analyze_reddit_posts")

graph_builder.add_edge("analyze_reddit_posts", "retrieve_reddit_post_data")


graph_builder.add_edge("retrieve_reddit_post_data", "analyze_google_results")
graph_builder.add_edge("retrieve_reddit_post_data", "analyze_bing_results")
graph_builder.add_edge("retrieve_reddit_post_data", "analyze_reddit_results")

'''concurently analyze all three sources and then move to final synthesis step'''
graph_builder.add_edge("analyze_google_results", "synthesize_final_answer")
graph_builder.add_edge("analyze_bing_results", "synthesize_final_answer")
graph_builder.add_edge("analyze_reddit_results", "synthesize_final_answer")

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