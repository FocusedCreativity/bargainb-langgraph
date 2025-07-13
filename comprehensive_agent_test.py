#!/usr/bin/env python3
"""
Comprehensive BargainB Memory Agent Test

Tests all major features:
1. Normal conversation responses
2. Memory functionality (storing and retrieving preferences)
3. Product search and recommendations
4. Conversation summarization
5. End-to-end user interaction flows
"""

import os
import sys
import asyncio
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from my_agent.memory_agent.agent import create_bargainb_memory_agent


def print_test_header(test_name: str):
    """Print a formatted test header"""
    print(f"\n{'='*60}")
    print(f"🧪 {test_name}")
    print(f"{'='*60}")


def print_test_result(test_name: str, success: bool, details: str = ""):
    """Print formatted test results"""
    status = "✅ PASSED" if success else "❌ FAILED"
    print(f"\n{status}: {test_name}")
    if details:
        print(f"Details: {details}")
    print("-" * 50)


async def test_normal_conversation():
    """Test basic conversation capabilities"""
    print_test_header("Normal Conversation Test")
    
    try:
        # Create agent
        app = create_bargainb_memory_agent()
        
        # Test configuration
        config = {
            "configurable": {
                "thread_id": "test_conversation",
                "user_id": "test_user_conv"
            }
        }
        
        # Normal conversation queries
        test_queries = [
            "Hello! How are you today?",
            "What's the weather like?",
            "Can you tell me about yourself?",
            "Thanks for your help!"
        ]
        
        all_passed = True
        
        for query in test_queries:
            print(f"\n👤 User: {query}")
            
            # Send query to agent
            result = await app.ainvoke(
                {"messages": [HumanMessage(content=query)]},
                config=config
            )
            
            # Get response
            if result and "messages" in result:
                last_message = result["messages"][-1]
                response = last_message.content if hasattr(last_message, 'content') else str(last_message)
                print(f"🐝 Beeb: {response}")
                
                # Check if response is reasonable (not error)
                if "error" in response.lower() or "failed" in response.lower():
                    all_passed = False
                    break
            else:
                all_passed = False
                break
        
        print_test_result("Normal Conversation", all_passed, 
                         "Agent responds to general queries without errors")
        return all_passed
        
    except Exception as e:
        print_test_result("Normal Conversation", False, f"Error: {str(e)}")
        return False


async def test_memory_functionality():
    """Test memory storage and retrieval"""
    print_test_header("Memory Functionality Test")
    
    try:
        # Create agent
        app = create_bargainb_memory_agent()
        
        # Test configuration
        config = {
            "configurable": {
                "thread_id": "test_memory",
                "user_id": "test_user_memory"
            }
        }
        
        # Test memory storage
        memory_queries = [
            "I'm vegetarian and I love organic food",
            "I'm on a tight budget, so I need cheap options",
            "I prefer shopping at Albert Heijn",
            "I have a gluten allergy"
        ]
        
        print("📝 Testing Memory Storage...")
        for query in memory_queries:
            print(f"\n👤 User: {query}")
            
            result = await app.ainvoke(
                {"messages": [HumanMessage(content=query)]},
                config=config
            )
            
            if result and "messages" in result:
                last_message = result["messages"][-1]
                response = last_message.content if hasattr(last_message, 'content') else str(last_message)
                print(f"🐝 Beeb: {response}")
        
        # Test memory retrieval
        print("\n🔍 Testing Memory Retrieval...")
        recall_query = "What do you remember about my dietary preferences and shopping habits?"
        print(f"\n👤 User: {recall_query}")
        
        result = await app.ainvoke(
            {"messages": [HumanMessage(content=recall_query)]},
            config=config
        )
        
        if result and "messages" in result:
            last_message = result["messages"][-1]
            response = last_message.content if hasattr(last_message, 'content') else str(last_message)
            print(f"🐝 Beeb: {response}")
            
            # Check if response contains memory details
            memory_indicators = ["vegetarian", "organic", "budget", "albert heijn", "gluten"]
            memory_recalled = any(indicator in response.lower() for indicator in memory_indicators)
            
            print_test_result("Memory Functionality", memory_recalled,
                             "Agent stores and recalls user preferences")
            return memory_recalled
        else:
            print_test_result("Memory Functionality", False, "No response from agent")
            return False
        
    except Exception as e:
        print_test_result("Memory Functionality", False, f"Error: {str(e)}")
        return False


async def test_product_search():
    """Test product search functionality"""
    print_test_header("Product Search Test")
    
    try:
        # Create agent
        app = create_bargainb_memory_agent()
        
        # Test configuration
        config = {
            "configurable": {
                "thread_id": "test_products",
                "user_id": "test_user_products"
            }
        }
        
        # Product search queries
        product_queries = [
            "Find me some organic milk",
            "I need cheap pasta for dinner",
            "Show me vegetarian protein options",
            "Compare prices for bread"
        ]
        
        all_passed = True
        
        for query in product_queries:
            print(f"\n👤 User: {query}")
            
            result = await app.ainvoke(
                {"messages": [HumanMessage(content=query)]},
                config=config
            )
            
            if result and "messages" in result:
                last_message = result["messages"][-1]
                response = last_message.content if hasattr(last_message, 'content') else str(last_message)
                print(f"🐝 Beeb: {response}")
                
                # Check if response contains product information
                product_indicators = ["€", "store", "price", "found", "product"]
                has_products = any(indicator in response.lower() for indicator in product_indicators)
                
                if not has_products:
                    all_passed = False
                    print(f"⚠️  No product information found in response")
            else:
                all_passed = False
                break
        
        print_test_result("Product Search", all_passed,
                         "Agent successfully searches and returns product information")
        return all_passed
        
    except Exception as e:
        print_test_result("Product Search", False, f"Error: {str(e)}")
        return False


async def test_summarization():
    """Test conversation summarization"""
    print_test_header("Conversation Summarization Test")
    
    try:
        # Create agent
        app = create_bargainb_memory_agent()
        
        # Test configuration
        config = {
            "configurable": {
                "thread_id": "test_summarization",
                "user_id": "test_user_summary"
            }
        }
        
        # Create a long conversation to trigger summarization
        long_conversation = [
            "Hello, I'm looking for healthy breakfast options",
            "I'm vegetarian and prefer organic foods",
            "Show me some cereals under €5",
            "That's too expensive, I need cheaper options",
            "I shop at Jumbo and Albert Heijn usually",
            "Do you have any gluten-free options?",
            "I need something quick for busy mornings",
            "What about oatmeal or muesli?",
            "I want to compare prices across different stores",
            "Can you recommend some protein-rich breakfast foods?"
        ]
        
        print("📝 Creating long conversation to trigger summarization...")
        
        # Process each message in the conversation
        for i, query in enumerate(long_conversation):
            print(f"\n👤 User ({i+1}/10): {query}")
            
            result = await app.ainvoke(
                {"messages": [HumanMessage(content=query)]},
                config=config
            )
            
            if result and "messages" in result:
                last_message = result["messages"][-1]
                response = last_message.content if hasattr(last_message, 'content') else str(last_message)
                print(f"🐝 Beeb: {response[:100]}...")
                
                # Check for summarization indicators
                if "summary" in response.lower() or "scribe bee" in response.lower():
                    print("✅ Summarization detected!")
                    print_test_result("Conversation Summarization", True,
                                     "Agent automatically summarizes long conversations")
                    return True
        
        # If no automatic summarization, test manual request
        print("\n🔍 Testing manual summarization request...")
        summary_query = "Can you summarize our conversation so far?"
        print(f"\n👤 User: {summary_query}")
        
        result = await app.ainvoke(
            {"messages": [HumanMessage(content=summary_query)]},
            config=config
        )
        
        if result and "messages" in result:
            last_message = result["messages"][-1]
            response = last_message.content if hasattr(last_message, 'content') else str(last_message)
            print(f"🐝 Beeb: {response}")
            
            # Check if response contains summary
            summary_indicators = ["summary", "discussed", "conversation", "preferences"]
            has_summary = any(indicator in response.lower() for indicator in summary_indicators)
            
            print_test_result("Conversation Summarization", has_summary,
                             "Agent provides conversation summaries when requested")
            return has_summary
        else:
            print_test_result("Conversation Summarization", False, "No response from agent")
            return False
        
    except Exception as e:
        print_test_result("Conversation Summarization", False, f"Error: {str(e)}")
        return False


async def test_end_to_end_flow():
    """Test complete user interaction flow"""
    print_test_header("End-to-End User Flow Test")
    
    try:
        # Create agent
        app = create_bargainb_memory_agent()
        
        # Test configuration
        config = {
            "configurable": {
                "thread_id": "test_e2e",
                "user_id": "test_user_e2e"
            }
        }
        
        # Complete user flow
        flow_queries = [
            "Hi! I'm new here. I'm vegetarian and love organic food",
            "I'm looking for healthy breakfast options under €3",
            "Which store has the best prices for organic cereals?",
            "Remember that I prefer Albert Heijn for shopping",
            "What do you remember about my preferences so far?"
        ]
        
        print("🔄 Testing complete user interaction flow...")
        
        for i, query in enumerate(flow_queries):
            print(f"\n👤 User (Step {i+1}): {query}")
            
            result = await app.ainvoke(
                {"messages": [HumanMessage(content=query)]},
                config=config
            )
            
            if result and "messages" in result:
                last_message = result["messages"][-1]
                response = last_message.content if hasattr(last_message, 'content') else str(last_message)
                print(f"🐝 Beeb: {response}")
                
                # Basic response validation
                if len(response) < 10 or "error" in response.lower():
                    print_test_result("End-to-End Flow", False, "Invalid response received")
                    return False
        
        print_test_result("End-to-End Flow", True,
                         "Complete user flow works from introduction to personalized responses")
        return True
        
    except Exception as e:
        print_test_result("End-to-End Flow", False, f"Error: {str(e)}")
        return False


async def run_comprehensive_test():
    """Run all tests and provide summary"""
    print("🚀 Starting Comprehensive BargainB Memory Agent Test")
    print("=" * 60)
    
    # Run all tests
    test_results = {
        "Normal Conversation": await test_normal_conversation(),
        "Memory Functionality": await test_memory_functionality(), 
        "Product Search": await test_product_search(),
        "Conversation Summarization": await test_summarization(),
        "End-to-End Flow": await test_end_to_end_flow()
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\n📈 Overall Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! The BargainB Memory Agent is working perfectly!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the results above.")
    
    return passed == total


if __name__ == "__main__":
    asyncio.run(run_comprehensive_test()) 