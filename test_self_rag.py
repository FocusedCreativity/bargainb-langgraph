#!/usr/bin/env python3
"""
Test script for the BargainB Self-RAG implementation.
"""

import asyncio
import os
from langchain_core.messages import HumanMessage
from my_agent.agent import graph

async def test_self_rag():
    """Test the Self-RAG system with grocery shopping questions."""
    
    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable is not set")
        print("Please set it using: export OPENAI_API_KEY='your-api-key-here'")
        return
    
    if not os.getenv("LANGSMITH_API_KEY"):
        print("⚠️  Warning: LANGSMITH_API_KEY environment variable is not set")
        print("LangSmith tracing will not be available")
    else:
        print("✅ LangSmith tracing is enabled")
    
    print("🛒 Testing BargainB Self-RAG Shopping Assistant")
    print("=" * 50)
    
    # Test question about organic milk
    question = "I'm looking for organic milk. What options do you have?"
    
    print(f"Question: {question}")
    print("-" * 50)
    
    # Create initial state
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "documents": [],
        "question": question,  # Set the question directly
        "generation": "",
        "generation_v_question_grade": "",
        "generation_v_documents_grade": "",
    }
    
    print("🔄 Running Self-RAG workflow...")
    print()
    
    try:
        # Run the graph
        result = await graph.ainvoke(initial_state)
        
        print("🎯 Final Results:")
        print("-" * 50)
        print(f"Question: {result['question']}")
        print(f"Number of products found: {len(result['documents'])}")
        print(f"Generation quality (vs product data): {result['generation_v_documents_grade']}")
        print(f"Generation quality (vs question): {result['generation_v_question_grade']}")
        print()
        
        # Show product details
        if result['documents']:
            print("📦 Found Products:")
            for i, doc in enumerate(result['documents'][:3], 1):  # Show first 3 products
                metadata = doc.metadata
                
                # Format price information
                price_display = "Unknown"
                if metadata.get('min_price') and metadata.get('max_price'):
                    min_p = metadata['min_price']
                    max_p = metadata['max_price'] 
                    if min_p == max_p:
                        price_display = f"€{min_p:.2f}"
                    else:
                        price_display = f"€{min_p:.2f} - €{max_p:.2f}"
                
                # Format store information
                store_display = "Unknown"
                if metadata.get('available_stores'):
                    store_count = metadata['available_stores']
                    store_names = metadata.get('store_names', '')
                    if store_names:
                        store_display = f"{store_count} stores ({store_names})"
                    else:
                        store_display = f"{store_count} stores"
                
                print(f"  {i}. {metadata.get('title', 'Unknown Product')}")
                print(f"     Brand: {metadata.get('brand', 'Unknown')}")
                print(f"     Price: {price_display}")
                print(f"     Stores: {store_display}")
                print()
        
        print("🤖 BargainB Response:")
        print("-" * 50)
        print(result['generation'])
        print()
        
        # Test with a different question
        print("\n" + "=" * 50)
        print("🛒 Testing with another shopping question...")
        print("=" * 50)
        
        question2 = "What healthy breakfast options do you have under €10?"
        
        print(f"Question: {question2}")
        print("-" * 50)
        
        # Create initial state for second question
        initial_state2 = {
            "messages": [HumanMessage(content=question2)],
            "documents": [],
            "question": question2,  # Set the question directly
            "generation": "",
            "generation_v_question_grade": "",
            "generation_v_documents_grade": "",
        }
        
        print("🔄 Running Self-RAG workflow...")
        print()
        
        # Run the graph
        result2 = await graph.ainvoke(initial_state2)
        
        print("🎯 Final Results:")
        print("-" * 50)
        print(f"Question: {result2['question']}")
        print(f"Number of products found: {len(result2['documents'])}")
        print(f"Generation quality (vs product data): {result2['generation_v_documents_grade']}")
        print(f"Generation quality (vs question): {result2['generation_v_question_grade']}")
        print()
        
        # Show product details
        if result2['documents']:
            print("📦 Found Products:")
            for i, doc in enumerate(result2['documents'][:3], 1):  # Show first 3 products
                metadata = doc.metadata
                
                # Format price information
                price_display = "Unknown"
                if metadata.get('min_price') and metadata.get('max_price'):
                    min_p = metadata['min_price']
                    max_p = metadata['max_price'] 
                    if min_p == max_p:
                        price_display = f"€{min_p:.2f}"
                    else:
                        price_display = f"€{min_p:.2f} - €{max_p:.2f}"
                elif metadata.get('pricing'):  # For category search results
                    pricing = metadata['pricing']
                    min_p = pricing.get('min_price')
                    max_p = pricing.get('max_price')
                    if min_p and max_p:
                        if min_p == max_p:
                            price_display = f"€{min_p:.2f}"
                        else:
                            price_display = f"€{min_p:.2f} - €{max_p:.2f}"
                
                # Format store information
                store_display = "Unknown"
                if metadata.get('available_stores'):
                    store_count = metadata['available_stores']
                    store_names = metadata.get('store_names', '')
                    if store_names:
                        store_display = f"{store_count} stores ({store_names})"
                    else:
                        store_display = f"{store_count} stores"
                elif metadata.get('pricing'):  # For category search results
                    pricing = metadata['pricing']
                    store_count = pricing.get('available_stores', 0)
                    store_names = pricing.get('store_names', [])
                    if store_count:
                        if store_names:
                            store_display = f"{store_count} stores ({', '.join(store_names)})"
                        else:
                            store_display = f"{store_count} stores"
                
                print(f"  {i}. {metadata.get('title', 'Unknown Product')}")
                print(f"     Brand: {metadata.get('brand', 'Unknown')}")
                print(f"     Price: {price_display}")
                print(f"     Stores: {store_display}")
                print()
        
        print("🤖 BargainB Response:")
        print("-" * 50)
        print(result2['generation'])
        
        # Test with a third question about specific categories
        print("\n" + "=" * 50)
        print("🛒 Testing category-based search...")
        print("=" * 50)
        
        question3 = "Show me some good cheese options for a party"
        
        print(f"Question: {question3}")
        print("-" * 50)
        
        # Create initial state for third question
        initial_state3 = {
            "messages": [HumanMessage(content=question3)],
            "documents": [],
            "question": question3,  # Set the question directly
            "generation": "",
            "generation_v_question_grade": "",
            "generation_v_documents_grade": "",
        }
        
        print("🔄 Running Self-RAG workflow...")
        print()
        
        # Run the graph
        result3 = await graph.ainvoke(initial_state3)
        
        print("🎯 Final Results:")
        print("-" * 50)
        print(f"Question: {result3['question']}")
        print(f"Number of products found: {len(result3['documents'])}")
        print(f"Generation quality (vs product data): {result3['generation_v_documents_grade']}")
        print(f"Generation quality (vs question): {result3['generation_v_question_grade']}")
        print()
        
        print("🤖 BargainB Response:")
        print("-" * 50)
        print(result3['generation'])
        
    except Exception as e:
        print(f"❌ Error during execution: {str(e)}")
        print("Please check your database connection and API keys")

if __name__ == "__main__":
    asyncio.run(test_self_rag()) 