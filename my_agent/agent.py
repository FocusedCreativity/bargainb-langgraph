"""Graph construction for the Self-RAG assistant."""
from langgraph.graph import StateGraph, START, END

from my_agent.utils.state import AgentState
from my_agent.utils.nodes import (
    retrieve_node,
    generate_node,
    grade_documents_node,
    transform_query_node,
    generate_generation_v_documents_grade_node,
    generate_generation_v_question_grade_node,
    decide_to_generate,
    grade_generation_v_documents,
    grade_generation_v_question,
)

# Build the Self-RAG graph
graph_builder = StateGraph(AgentState)

# Add nodes for the Self-RAG workflow
graph_builder.add_node("retrieve", retrieve_node)
graph_builder.add_node("grade_documents", grade_documents_node)
graph_builder.add_node("generate", generate_node)
graph_builder.add_node("transform_query", transform_query_node)
graph_builder.add_node("generate_generation_v_documents_grade", generate_generation_v_documents_grade_node)
graph_builder.add_node("generate_generation_v_question_grade", generate_generation_v_question_grade_node)

# Define edges and conditional routes
graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "grade_documents")

# Conditional edge: decide whether to generate or transform query
graph_builder.add_conditional_edges(
    "grade_documents", 
    decide_to_generate, 
    {
        "transform_query": "transform_query",
        "generate": "generate",
    }
)

# Transform query loops back to retrieve
graph_builder.add_edge("transform_query", "retrieve")

# Generate goes to document grading
graph_builder.add_edge("generate", "generate_generation_v_documents_grade")

# Conditional edge: check if generation is supported by documents
graph_builder.add_conditional_edges(
    "generate_generation_v_documents_grade",
    grade_generation_v_documents,
    {
        "supported": "generate_generation_v_question_grade",
        "not_supported": "generate",
    }
)

# Conditional edge: check if generation is useful for the question
graph_builder.add_conditional_edges(
    "generate_generation_v_question_grade",
    grade_generation_v_question,
    {
        "useful": END,
        "not_useful": "transform_query",
    }
)

# Compile the graph
graph = graph_builder.compile()
