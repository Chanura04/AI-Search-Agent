# AI Search Agent

This AI Search Agent is a sophisticated research assistant that aggregates and synthesizes real-time information from Google, Bing, and Reddit to answer complex user queries. Built on LangGraph, it orchestrates parallel search workflows and employs Llama 3.1 (via NVIDIA NIM) to reason through data and generate comprehensive, cited responses. The system features deep Reddit integration to retrieve and analyze community discussions, ensuring a balance of factual data and human insights. Users can interact via a responsive Streamlit web interface , with all search data efficiently cached in MongoDB for persistence and performance.

## 🌟 Features

- **Parallel Multi-Source Search**: Simultaneously queries Google, Bing, and Reddit.
- **Deep Reddit Integration**:
  - Searches for relevant discussions.
  - Filters for high-value threads using AI.
  - Retrieves and analyzes full comment trees for community insights.
- **AI Synthesis**: Uses Llama 3.1 to analyze each source independently and synthesize a comprehensive final answer.
- **Dual Interfaces**:
  - **Web UI**: Built with Streamlit for an interactive chat experience.
  - **CLI**: Terminal-based interaction for quick lookups.
- **Data Persistence**: Caches search results and comments in MongoDB.

## 💻 Technologies

- **Language**: Python 3.10+
- **Orchestration**: LangGraph, LangChain
- **AI Model**: Llama 3.1 (via NVIDIA NIM)
- **Data Retrieval**: BrightData (SERP API & Datasets)
- **Database**: MongoDB
- **Frontend**: Streamlit

## 📋 Prerequisites

- **Python 3.10+**
- **MongoDB**: Must be running locally on port `27017`.
- **API Keys**:
  - BrightData (for SERP API and Datasets).
  - NVIDIA NIM (for Llama 3.1).

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Chanura04/AI-Search-Agent.git
   cd AI-Search-Agent
   ```

2. **Install dependencies**
   ```bash
   pip install langchain langgraph streamlit pymongo python-dotenv requests pydantic langchain-nvidia-ai-endpoints
   ```

3. **Configuration**
   Create a `.env` file in the root directory:
   ```env
   BRIGHTDATA_API_KEY=your_brightdata_api_key
   NVIDIA_API_KEY=your_nvidia_api_key
   ```

   *Note: You may need to update the BrightData Zone name (`ai_agent`) and Dataset IDs in `web_operations.py` to match your account configuration.*

4. **Start MongoDB**
   Ensure your local MongoDB instance is running:
   ```bash
   mongod
   ```

## 🚀 Usage

### Web Interface
Launch the Streamlit application:
   ```bash
   streamlit run app.py
   ```

### Command Line Interface
Run the agent in the terminal:
   ```bash
   python main.py
   ```

## 📂 Project Structure

- **`app.py`**: Streamlit frontend application.
- **`main.py`**: CLI entry point and LangGraph workflow definition.
- **`web_operations.py`**: Handles BrightData API requests (SERP & Datasets) and MongoDB operations.
- **`snapshot_operations.py`**: Utilities for managing asynchronous BrightData snapshots.
- **`prompts.py`**: Contains all LLM prompt templates (Analysis, Synthesis, etc.).
- **`test.py`**: Connection testing scripts.

## 🧠 How It Works

1. **Search**: The agent triggers parallel searches on Google, Bing, and Reddit.
2. **Selection**: It analyzes Reddit search results to pick the most relevant threads.
3. **Retrieval**: Fetches full comments from selected Reddit posts.
4. **Synthesis**: Analyzes results from all sources and combines them into a final, cited response.