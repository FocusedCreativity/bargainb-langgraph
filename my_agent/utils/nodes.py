import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, AIMessage
from pydantic import BaseModel, Field
from typing import List, Literal
import logging

# Set up LangSmith tracing if API key is available
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "self-rag"  # Optional: project name for LangSmith

from .state import AgentState
from .tools import retriever, format_documents

# Initialize the LLM
model = ChatOpenAI(model="gpt-4o", temperature=0)

# Pydantic models for structured output
class GradeDocuments(BaseModel):
    """Binary score for document relevance."""
    binary_score: Literal["yes", "no"] = Field(
        description="Relevance score 'yes' or 'no'"
    )

class GradeGeneration(BaseModel):
    """Binary score for generation quality."""
    binary_score: Literal["yes", "no"] = Field(
        description="Binary score 'yes' or 'no'"
    )

async def retrieve_node(state: AgentState, config: RunnableConfig = None) -> AgentState:
    """Retrieve products from BargainB database relevant to the question."""
    print("---RETRIEVE FROM BARGAINB DATABASE---")
    
    question = state.get("question", "")
    if not question:
        # Extract question from the last message
        if state.get("messages"):
            question = state["messages"][-1].content
    
    # If this is the first retrieve and we have a full sentence, extract keywords
    if len(question.split()) > 3 and not state.get("documents"):
        print(f"🔧 Simplifying query from: '{question}'")
        # Simple keyword extraction for common shopping queries
        keywords = []
        query_lower = question.lower()
        
        # Extract product types
        product_types = ['milk', 'bread', 'cheese', 'meat', 'fish', 'fruit', 'vegetable', 'snack', 'drink', 'breakfast', 'lunch', 'dinner']
        for product in product_types:
            if product in query_lower:
                keywords.append(product)
        
        # Extract modifiers
        modifiers = ['organic', 'low-fat', 'whole', 'fresh', 'healthy', 'cheap', 'expensive', 'bio', 'natural']
        for modifier in modifiers:
            if modifier in query_lower:
                keywords.append(modifier)
        
        # If we found keywords, use them
        if keywords:
            question = ' '.join(keywords[:3])  # Max 3 keywords
            print(f"🔧 Simplified to: '{question}'")
    
    print(f"🔍 Searching for: '{question}'")
    
    # Retrieve products from BargainB database
    documents = await retriever.invoke(question)
    
    return {
        "documents": documents,
        "question": question,
        "messages": state.get("messages", []),
    }

async def generate_node(state: AgentState, config: RunnableConfig = None) -> AgentState:
    """Generate answer using retrieved product information."""
    print("---GENERATE PRODUCT RECOMMENDATION---")
    
    # Create product-focused RAG prompt
    prompt = ChatPromptTemplate.from_template(
        """You are BargainB, an intelligent grocery shopping assistant. 
        Use the following product information to answer the user's question about groceries, products, or shopping.
        
        Focus on:
        - Product recommendations based on the query
        - Price comparisons when available
        - Nutritional information when relevant
        - Store availability
        - Brand information
        
        If you can't find specific products, suggest alternatives or explain what's available.
        Keep your response helpful and shopping-focused. Always give a specific price for each store, never a range.

        User Question: {question}
        
        Available Products:
        {context}
        
        Response:"""
    )
    
    # Create RAG chain
    rag_chain = prompt | model | StrOutputParser()
    
    # Generate answer
    generation = await rag_chain.ainvoke({
        "context": format_documents(state.get("documents", [])),
        "question": state.get("question", ""),
    })
    
    return {
        "generation": generation,
        "documents": state.get("documents", []),
        "question": state.get("question", ""),
        "messages": state.get("messages", []),
    }

async def grade_documents_node(state: AgentState, config: RunnableConfig = None) -> AgentState:
    """Grade the relevance of retrieved products to the user's question."""
    print("---CHECK PRODUCT RELEVANCE---")
    
    # Create grading prompt
    prompt = ChatPromptTemplate.from_template(
        """You are a grader assessing whether a retrieved product is relevant to a user's grocery/shopping question.
        
        Here is the product information:
        {context}

        Here is the user's question: {question}

        Consider the product relevant if it:
        - Matches the product type mentioned in the question
        - Belongs to a category related to the question
        - Has ingredients or features mentioned in the question
        - Could be a reasonable alternative to what was asked
        
        Give a binary score 'yes' or 'no' to indicate whether the product is relevant to the question."""
    )
    
    # Create grading chain
    grader = prompt | model.with_structured_output(GradeDocuments)
    
    # Grade each document
    filtered_docs = []
    for doc in state.get("documents", []):
        grade = await grader.ainvoke({
            "context": doc.page_content,
            "question": state.get("question", ""),
        })
        
        if grade.binary_score == "yes":
            print(f"---GRADE: PRODUCT RELEVANT - {doc.metadata.get('title', 'Unknown Product')}---")
            filtered_docs.append(doc)
        else:
            print(f"---GRADE: PRODUCT NOT RELEVANT - {doc.metadata.get('title', 'Unknown Product')}---")
    
    return {
        "documents": filtered_docs,
        "question": state.get("question", ""),
        "messages": state.get("messages", []),
    }

async def transform_query_node(state: AgentState, config: RunnableConfig = None) -> AgentState:
    """Transform the query to produce a better product search."""
    print("---TRANSFORM PRODUCT QUERY---")
    
    # Create query transformation prompt
    prompt = ChatPromptTemplate.from_template(
        """You are optimizing a grocery/product search query for better results.
        
        Transform the user's question into a simple, direct product search query that will find relevant groceries.
        
        Rules:
        - Use simple keywords, not full sentences
        - Focus on the main product type (e.g., "milk", "bread", "cheese")
        - Include key modifiers (e.g., "organic", "low-fat", "whole wheat")
        - Keep it under 5 words
        - Do not use quotes or complex phrases
        
        Examples:
        - "I'm looking for organic milk" → "organic milk"
        - "What healthy breakfast options do you have?" → "healthy breakfast"
        - "Show me some cheese for a party" → "cheese"
        
        Original question: {question}
        
        Improved search query:"""
    )
    
    # Create transformation chain
    chain = prompt | model | StrOutputParser()
    
    # Transform the question
    better_question = await chain.ainvoke({"question": state.get("question", "")})
    
    return {
        "question": better_question,
        "documents": state.get("documents", []),
        "generation": state.get("generation", ""),
        "messages": state.get("messages", []),
    }

async def generate_generation_v_documents_grade_node(state: AgentState, config: RunnableConfig = None) -> AgentState:
    """Grade whether the generation is grounded in the retrieved product information."""
    print("---GRADE GENERATION vs PRODUCT DATA---")
    
    prompt = ChatPromptTemplate.from_template(
        """You are grading whether a response about products is supported by the provided product information.
        
        Here is the product information:
        {documents}
        
        Here is the response: {generation}
        
        Check if the response:
        - Only mentions products that are in the provided data
        - Accurately represents prices, brands, and features
        - Doesn't invent product details not in the data
        - Bases recommendations on actual available products
        
        Give a binary score 'yes' or 'no' to indicate whether the response is grounded in the product data."""
    )
    
    # Create grading chain
    grader = prompt | model.with_structured_output(GradeGeneration)
    
    # Grade the generation
    score = await grader.ainvoke({
        "documents": format_documents(state.get("documents", [])),
        "generation": state.get("generation", ""),
    })
    
    return {
        "generation_v_documents_grade": score.binary_score,
        "documents": state.get("documents", []),
        "question": state.get("question", ""),
        "generation": state.get("generation", ""),
        "messages": state.get("messages", []),
    }

async def generate_generation_v_question_grade_node(state: AgentState, config: RunnableConfig = None) -> AgentState:
    """Grade whether the generation addresses the user's shopping question."""
    print("---GRADE GENERATION vs SHOPPING QUESTION---")
    
    prompt = ChatPromptTemplate.from_template(
        """You are grading whether a response is useful for answering a shopping/grocery question.
        
        Here is the response:
        {generation}
        
        Here is the user's question: {question}
        
        Check if the response:
        - Directly addresses what the user was looking for
        - Provides helpful product recommendations
        - Includes relevant details like prices or stores
        - Offers alternatives if exact matches aren't available
        
        Give a binary score 'yes' or 'no' to indicate whether the response is useful for the shopping question."""
    )
    
    # Create grading chain
    grader = prompt | model.with_structured_output(GradeGeneration)
    
    # Grade the generation
    score = await grader.ainvoke({
        "question": state.get("question", ""),
        "generation": state.get("generation", ""),
    })
    
    return {
        "generation_v_question_grade": score.binary_score,
        "documents": state.get("documents", []),
        "question": state.get("question", ""),
        "generation": state.get("generation", ""),
        "messages": state.get("messages", []),
    }

# Routing functions
def decide_to_generate(state: AgentState) -> Literal["transform_query", "generate"]:
    """Decide whether to generate an answer or transform the query."""
    print("---DECIDE TO GENERATE---")
    
    documents = state.get("documents", [])
    if not documents:
        print("---DECISION: TRANSFORM QUERY - No relevant products found---")
        return "transform_query"
    else:
        print(f"---DECISION: GENERATE - Found {len(documents)} relevant products---")
        return "generate"

def grade_generation_v_documents(state: AgentState) -> Literal["supported", "not_supported"]:
    """Route based on generation vs documents grade."""
    print("---GRADE GENERATION vs PRODUCT DATA---")
    
    if state.get("generation_v_documents_grade") == "yes":
        print("---DECISION: SUPPORTED - Response is grounded in product data---")
        return "supported"
    else:
        print("---DECISION: NOT SUPPORTED - Response not grounded in product data---")
        return "not_supported"

def grade_generation_v_question(state: AgentState) -> Literal["useful", "not_useful"]:
    """Route based on generation vs question grade."""
    print("---GRADE GENERATION vs SHOPPING QUESTION---")
    
    if state.get("generation_v_question_grade") == "yes":
        print("---DECISION: USEFUL - Response addresses shopping question---")
        return "useful"
    else:
        print("---DECISION: NOT USEFUL - Response doesn't address shopping question---")
        return "not_useful"
