# Self-RAG Implementation with LangGraph

This repository demonstrates a Self-RAG (Self-Reflective Retrieval-Augmented Generation) implementation using [LangGraph](https://github.com/langchain-ai/langgraph). The system performs self-reflection on retrieval quality and generation accuracy to provide better answers.

## Features

- **Self-Reflective Retrieval**: Automatically grades document relevance before generation
- **Generation Quality Assessment**: Evaluates if generated answers are grounded in retrieved documents
- **Query Transformation**: Improves queries when initial retrieval is inadequate
- **Answer Usefulness Validation**: Ensures generated answers actually address the user's question
- **Iterative Improvement**: Loops back to improve retrieval or generation when quality is poor

## How It Works

The Self-RAG system implements a multi-step workflow:

1. **Retrieve**: Gets relevant documents for the user's question
2. **Grade Documents**: Evaluates if retrieved documents are relevant
3. **Transform Query** (if needed): Improves the query for better retrieval
4. **Generate**: Creates an answer using retrieved documents
5. **Grade Generation vs Documents**: Checks if the answer is grounded in facts
6. **Grade Generation vs Question**: Validates if the answer addresses the question
7. **Iterate**: Loops back to improve retrieval or generation if quality is poor

## Setup

### Prerequisites

1. Install the dependencies:

```bash
pip install -r requirements.txt
```

### Environment Variables

Set up the following environment variables:

```bash
# Required: OpenAI API Key
export OPENAI_API_KEY="your-openai-api-key-here"

# Optional: LangSmith API Key for tracing and debugging
export LANGSMITH_API_KEY="your-langsmith-api-key-here"
```

The system will automatically:
- Use the OpenAI API key for LLM operations
- Enable LangSmith tracing if the API key is present
- Set up the project name as "self-rag" in LangSmith

### Verify Setup

Run the setup verification script to ensure everything is configured correctly:

```bash
python check_setup.py
```

This will:
- Check if all required environment variables are set
- Verify all dependencies are installed
- Test OpenAI API connection
- Test document loading functionality

## Running the System

### Quick Test

Run the included test script:

```bash
python test_self_rag.py
```

This will:
- Check for required environment variables
- Test the Self-RAG system with two sample questions
- Show the complete workflow with debugging output
- Display generation quality scores

### Using the Graph Directly

```python
from my_agent.agent import graph
from langchain_core.messages import HumanMessage

# Example usage
result = await graph.ainvoke({
    "messages": [HumanMessage(content="Explain how the different types of agent memory work.")],
    "documents": [],
    "question": "",
    "generation": "",
    "generation_v_question_grade": "",
    "generation_v_documents_grade": "",
})

print(result["generation"])
```

## Configuration

The system uses default URLs for document retrieval (LangChain blog posts about agents). You can customize the knowledge base by modifying the URLs in `my_agent/utils/tools.py`.

## Architecture

- **State Management**: `my_agent/utils/state.py` - Manages document state, questions, and generation quality grades
- **Graph Nodes**: `my_agent/utils/nodes.py` - Implements retrieval, generation, and grading logic
- **Tools**: `my_agent/utils/tools.py` - Handles document loading and vector store setup
- **Graph Definition**: `my_agent/agent.py` - Defines the Self-RAG workflow

## Troubleshooting

### Common Issues

1. **Missing OpenAI API Key**: Ensure `OPENAI_API_KEY` is set in your environment
2. **Network Issues**: Check your internet connection for document retrieval
3. **Import Errors**: Run `pip install -r requirements.txt` to install dependencies

### Debug Output

The system provides detailed console output showing:
- Document retrieval progress
- Relevance grading results
- Generation quality assessments
- Decision routing logic

### Setup Verification

If you encounter issues, run the setup verification script:

```bash
python check_setup.py
```

This will help identify and troubleshoot configuration problems.

## Resources

- [Self-RAG Paper](https://arxiv.org/abs/2310.11511)
- [LangGraph Documentation](https://github.com/langchain-ai/langgraph)
- [LangGraph Configuration Guide](https://langchain-ai.github.io/langgraph)
