import streamlit as st
from dotenv import load_dotenv
from typing import Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from web_operations import serp_search, reddit_search_api, reddit_post_retrieval
from prompts import (
    PromptTemplates,
    get_google_analysis_messages,
    get_bing_analysis_messages,
    get_reddit_analysis_messages,
    get_synthesis_messages,
    get_reddit_url_analysis_messages,
)

# Load environment variables
load_dotenv()

# Initialize LLM
@st.cache_resource
def get_llm():
    return init_chat_model("meta/llama-3.1-8b-instruct", model_provider="nvidia")


# Define State and Models
class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_qestion: str | None
    google_results: list | None
    bing_results: list | None
    reddit_results: list | None
    selected_reddut_urls: list[str] | None
    reddit_post_data: list | None
    google_analysis: str | None
    bing_analysis: str | None
    reddit_analysis: str | None
    final_answer: str | None


class RedditURLAnalysis(BaseModel):
    selected_urls: List[str] = Field(
        description="List of Reddit URLs that contain valuable information for answering the user's question"
    )


# Graph node functions
def google_search(state: State):
    user_question = state.get("user_qestion", "")
    google_result = serp_search(user_question, engine="google")
    return {"google_results": google_result}


def bing_search(state: State):
    user_question = state.get("user_qestion", "")
    bing_result = serp_search(user_question, engine="bing")
    return {"bing_results": bing_result}


def reddit_search(state: State):
    user_question = state.get("user_qestion", "")
    reddit_result = reddit_search_api(user_question)
    return {"reddit_results": reddit_result}


def analyze_reddit_posts(state: State):
    llm = get_llm()
    user_question = state.get("user_question", "")
    reddit_results = state.get("reddit_results", "")

    if not reddit_results:
        return {"selected_reddit_urls": []}

    structured_llm = llm.with_structured_output(RedditURLAnalysis)
    messages = get_reddit_url_analysis_messages(user_question, reddit_results)

    try:
        analysis = structured_llm.invoke(messages)
        selected_urls = analysis.selected_urls
    except Exception as e:
        selected_urls = []

    return {"selected_reddit_urls": selected_urls}


def retrieve_reddit_post_data(state: State):
    selected_urls = state.get("selected_reddit_urls", [])

    if not selected_urls:
        return {"reddit_post_data": []}

    reddit_post_data = reddit_post_retrieval(selected_urls)

    if not reddit_post_data:
        reddit_post_data = []

    return {"reddit_post_data": reddit_post_data}


def analyze_google_results(state: State):
    llm = get_llm()
    user_question = state.get("user_question", "")
    google_results = state.get("google_results", "")

    messages = get_google_analysis_messages(user_question, google_results)
    reply = llm.invoke(messages)

    return {"google_analysis": reply.content}


def analyze_bing_results(state: State):
    llm = get_llm()
    user_question = state.get("user_question", "")
    bing_results = state.get("bing_results", "")

    messages = get_bing_analysis_messages(user_question, bing_results)
    reply = llm.invoke(messages)

    return {"bing_analysis": reply.content}


def analyze_reddit_results(state: State):
    llm = get_llm()
    user_question = state.get("user_question", "")
    reddit_results = state.get("reddit_results", "")
    reddit_post_data = state.get("reddit_post_data", "")

    messages = get_reddit_analysis_messages(
        user_question, reddit_results, reddit_post_data
    )
    reply = llm.invoke(messages)

    return {"reddit_analysis": reply.content}


def synthesize_analyses(state: State):
    llm = get_llm()
    user_question = state.get("user_question", "")
    google_analysis = state.get("google_analysis", "")
    bing_analysis = state.get("bing_analysis", "")
    reddit_analysis = state.get("reddit_analysis", "")

    messages = get_synthesis_messages(
        user_question, google_analysis, bing_analysis, reddit_analysis
    )

    reply = llm.invoke(messages)
    final_answer = reply.content

    return {
        "final_answer": final_answer,
        "messages": [{"role": "assistant", "content": final_answer}],
    }


# Build graph
@st.cache_resource
def build_graph():
    graph_builder = StateGraph(State)

    # Add nodes
    graph_builder.add_node("google_search", google_search)
    graph_builder.add_node("bing_search", bing_search)
    graph_builder.add_node("reddit_search", reddit_search)
    graph_builder.add_node("analyze_reddit_posts", analyze_reddit_posts)
    graph_builder.add_node("retrieve_reddit_post_data", retrieve_reddit_post_data)
    graph_builder.add_node("analyze_google_results", analyze_google_results)
    graph_builder.add_node("analyze_bing_results", analyze_bing_results)
    graph_builder.add_node("analyze_reddit_results", analyze_reddit_results)
    graph_builder.add_node("synthesize_final_answer", synthesize_analyses)

    # Connect nodes - parallel search
    graph_builder.add_edge(START, "google_search")
    graph_builder.add_edge(START, "bing_search")
    graph_builder.add_edge(START, "reddit_search")

    # Reddit post analysis
    graph_builder.add_edge("google_search", "analyze_reddit_posts")
    graph_builder.add_edge("bing_search", "analyze_reddit_posts")
    graph_builder.add_edge("reddit_search", "analyze_reddit_posts")
    graph_builder.add_edge("analyze_reddit_posts", "retrieve_reddit_post_data")

    # Analyze results
    graph_builder.add_edge("retrieve_reddit_post_data", "analyze_google_results")
    graph_builder.add_edge("retrieve_reddit_post_data", "analyze_bing_results")
    graph_builder.add_edge("retrieve_reddit_post_data", "analyze_reddit_results")

    # Synthesize final answer
    graph_builder.add_edge("analyze_google_results", "synthesize_final_answer")
    graph_builder.add_edge("analyze_bing_results", "synthesize_final_answer")
    graph_builder.add_edge("analyze_reddit_results", "synthesize_final_answer")

    graph_builder.add_edge("synthesize_final_answer", END)

    return graph_builder.compile()


# Streamlit UI
def main():
    st.set_page_config(
        page_title="AI Search Agent",
        page_icon="🔍",
        layout="wide",
    )

    st.title("🔍 AI Search Agent")
    st.markdown(
        "Ask any question and get comprehensive answers from Google, Bing, and Reddit!"
    )

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if user_query := st.chat_input("Ask a question..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Display assistant response with progress
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching Google, Bing, and Reddit..."):
                # Create status placeholder
                status_placeholder = st.empty()
                
                # Initialize state
                state = {
                    "messages": [{"role": "user", "content": user_query}],
                    "user_qestion": user_query,
                    "google_results": None,
                    "bing_results": None,
                    "reddit_results": None,
                    "google_analysis": None,
                    "bing_analysis": None,
                    "selected_reddut_urls": None,
                    "reddit_post_data": None,
                    "reddit_analysis": None,
                    "final_answer": None,
                }

                # Show search progress
                status_placeholder.info("⏳ Performing parallel searches...")
                
                # Build and invoke graph
                graph = build_graph()
                final_state = graph.invoke(state)

                # Clear status
                status_placeholder.empty()

                # Display final answer
                if final_answer := final_state.get("final_answer"):
                    st.markdown(final_answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": final_answer}
                    )
                else:
                    error_msg = "Sorry, I couldn't generate an answer. Please try again."
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )

    # Sidebar with info
    with st.sidebar:
        st.header("About")
        st.markdown(
            """
            This AI Search Agent combines results from:
            - 🔍 **Google Search**
            - 🔍 **Bing Search**
            - 💬 **Reddit Discussions**
            
            It analyzes and synthesizes information from all sources 
            to provide comprehensive, well-rounded answers.
            """
        ) 
        
        st.divider()
        
        st.header("How it works")
        st.markdown(
            """
            1. **Parallel Search**: Searches Google, Bing, and Reddit simultaneously
            2. **Reddit Analysis**: Analyzes Reddit posts for relevant discussions
            3. **Source Analysis**: Analyzes results from each source
            4. **Synthesis**: Combines all insights into a comprehensive answer
            """
        )
        
        st.divider()
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()


if __name__ == "__main__":
    main()