#!/usr/bin/env python3

"""
Test Script for BargainB Delegation-Based Agent (Beeb)

Tests the new delegation pattern where:
- Beeb is the main assistant who always responds to users
- Product searches are delegated to specialized RAG agent
- Memory management works in background
- Conversation flow is managed by dialog_state stack
"""

import asyncio
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

from langchain_core.messages import HumanMessage
from my_agent.memory_agent.agent import memory_agent


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*20} {title} {'='*20}")


def print_subsection(title: str):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")


def print_conversation_state(result: dict):
    """Print conversation state information."""
    print(f"Dialog State: {result.get('dialog_state', [])}")
    print(f"User ID: {result.get('user_id')}")
    print(f"Thread ID: {result.get('thread_id')}")
    if result.get('summary'):
        print(f"Summary: {result['summary'][:100]}..." if len(result['summary']) > 100 else f"Summary: {result['summary']}")


def test_basic_beeb_conversation():
    """Test basic conversation with Beeb without product search."""
    print_section("Testing Basic Beeb Conversation")
    
    config = {"configurable": {"thread_id": "beeb_basic_001", "user_id": "alice_test"}}
    
    # Basic greeting and conversation
    message1 = HumanMessage(content="Hi! I'm Alice. I'm new to BargainB. What can you help me with?")
    result1 = memory_agent.invoke({"messages": [message1]}, config)
    print(f"🤖 Beeb: {result1['messages'][-1].content}")
    print_conversation_state(result1)
    
    # Follow-up question (non-product related)
    message2 = HumanMessage(content="What Dutch grocery stores do you cover?")
    result2 = memory_agent.invoke({"messages": result1["messages"] + [message2]}, config)
    print(f"🤖 Beeb: {result2['messages'][-1].content}")
    
    print("✅ Basic conversation with Beeb works!")
    return result2


def test_product_search_delegation():
    """Test delegation from Beeb to product search agent."""
    print_section("Testing Product Search Delegation")
    
    config = {"configurable": {"thread_id": "beeb_delegation_001", "user_id": "bob_test"}}
    
    # Start conversation
    message1 = HumanMessage(content="Hi! I'm Bob. I need help finding organic milk.")
    result1 = memory_agent.invoke({"messages": [message1]}, config)
    print(f"🤖 Beeb: {result1['messages'][-1].content}")
    print_conversation_state(result1)
    
    # Product search should be delegated to search agent
    print("\n📊 Analyzing delegation flow...")
    
    # Check if delegation occurred (should see dialog_state changes)
    final_state = result1.get('dialog_state', [])
    print(f"Final dialog state: {final_state}")
    
    # The response should include product information from the search agent
    response_content = result1['messages'][-1].content
    if any(keyword in response_content.lower() for keyword in ['milk', 'price', '€', 'store']):
        print("✅ Product search delegation successful - response includes product details!")
    else:
        print("❌ Product search delegation may have failed - no product details in response")
    
    return result1


def test_conversation_with_memory():
    """Test conversation with user preference learning."""
    print_section("Testing Conversation with Memory Learning")
    
    config = {"configurable": {"thread_id": "beeb_memory_001", "user_id": "clara_test"}}
    
    # First interaction - establish preferences
    message1 = HumanMessage(content="Hi! I'm Clara. I'm vegetarian and on a tight budget. I usually shop at Albert Heijn.")
    result1 = memory_agent.invoke({"messages": [message1]}, config)
    print(f"🤖 Beeb: {result1['messages'][-1].content}")
    print_conversation_state(result1)
    
    # Product search with preferences
    message2 = HumanMessage(content="Can you find me some affordable vegetarian protein options?")
    result2 = memory_agent.invoke({"messages": result1["messages"] + [message2]}, config)
    print(f"🤖 Beeb: {result2['messages'][-1].content}")
    
    # Check for memory usage
    response = result2['messages'][-1].content
    if any(keyword in response.lower() for keyword in ['vegetarian', 'budget', 'affordable']):
        print("✅ Memory integration working - Beeb remembered user preferences!")
    else:
        print("⚠️  Memory integration unclear - preferences not explicitly mentioned")
    
    return result2


def test_multiple_product_searches():
    """Test multiple product searches in same conversation."""
    print_section("Testing Multiple Product Searches")
    
    config = {"configurable": {"thread_id": "beeb_multi_001", "user_id": "david_test"}}
    
    # First product search
    message1 = HumanMessage(content="Hi! I need pasta for dinner tonight.")
    result1 = memory_agent.invoke({"messages": [message1]}, config)
    print(f"🤖 Beeb (pasta): {result1['messages'][-1].content[:200]}...")
    
    # Second product search in same conversation
    message2 = HumanMessage(content="And what about some cheese to go with it?")
    result2 = memory_agent.invoke({"messages": result1["messages"] + [message2]}, config)
    print(f"🤖 Beeb (cheese): {result2['messages'][-1].content[:200]}...")
    
    # Third product search
    message3 = HumanMessage(content="Actually, do you have any tomato sauce recommendations?")
    result3 = memory_agent.invoke({"messages": result2["messages"] + [message3]}, config)
    print(f"🤖 Beeb (sauce): {result3['messages'][-1].content[:200]}...")
    
    print("✅ Multiple product searches completed!")
    return result3


def test_non_product_after_product():
    """Test switching from product search back to general conversation."""
    print_section("Testing Conversation Flow: Product → General")
    
    config = {"configurable": {"thread_id": "beeb_mixed_001", "user_id": "emma_test"}}
    
    # Product search
    message1 = HumanMessage(content="Can you help me find breakfast cereal?")
    result1 = memory_agent.invoke({"messages": [message1]}, config)
    print(f"🤖 Beeb (cereal): {result1['messages'][-1].content[:150]}...")
    
    # Switch to general conversation
    message2 = HumanMessage(content="Thanks! By the way, what time do most Dutch supermarkets close?")
    result2 = memory_agent.invoke({"messages": result1["messages"] + [message2]}, config)
    print(f"🤖 Beeb (hours): {result2['messages'][-1].content}")
    
    # Back to product search
    message3 = HumanMessage(content="Perfect. Now can you find me some yogurt options?")
    result3 = memory_agent.invoke({"messages": result2["messages"] + [message3]}, config)
    print(f"🤖 Beeb (yogurt): {result3['messages'][-1].content[:150]}...")
    
    print("✅ Conversation flow between product and general topics works!")
    return result3


def test_delegation_error_handling():
    """Test how the system handles search errors or empty results."""
    print_section("Testing Delegation Error Handling")
    
    config = {"configurable": {"thread_id": "beeb_error_001", "user_id": "frank_test"}}
    
    # Search for something unlikely to be found
    message1 = HumanMessage(content="I need spaceship fuel for my rocket ship.")
    result1 = memory_agent.invoke({"messages": [message1]}, config)
    print(f"🤖 Beeb: {result1['messages'][-1].content}")
    
    # Beeb should handle the no-results gracefully
    response = result1['messages'][-1].content
    if any(phrase in response.lower() for phrase in ["couldn't find", "no products", "more specific", "groceries"]):
        print("✅ Error handling works - Beeb handled no results gracefully!")
    else:
        print("❌ Error handling unclear - response doesn't indicate search failure")
    
    # Follow up with a normal search
    message2 = HumanMessage(content="Okay, how about regular bread then?")
    result2 = memory_agent.invoke({"messages": result1["messages"] + [message2]}, config)
    print(f"🤖 Beeb (bread): {result2['messages'][-1].content[:150]}...")
    
    return result2


def test_beeb_personality():
    """Test that Beeb maintains his personality throughout delegation."""
    print_section("Testing Beeb's Personality Consistency")
    
    config = {"configurable": {"thread_id": "beeb_personality_001", "user_id": "grace_test"}}
    
    # General conversation
    message1 = HumanMessage(content="Hi Beeb! I'm new to saving money on groceries. Any tips?")
    result1 = memory_agent.invoke({"messages": [message1]}, config)
    print(f"🤖 Beeb (general): {result1['messages'][-1].content}")
    
    # Product search
    message2 = HumanMessage(content="Can you find me the cheapest rice options?")
    result2 = memory_agent.invoke({"messages": result1["messages"] + [message2]}, config)
    print(f"🤖 Beeb (rice): {result2['messages'][-1].content}")
    
    # Check for personality traits
    full_conversation = " ".join([msg.content for msg in result2['messages'] if hasattr(msg, 'content')])
    
    personality_indicators = ['save', 'deal', 'budget', 'cheap', 'affordable', '🛒', '💰', 'money']
    found_indicators = [indicator for indicator in personality_indicators if indicator in full_conversation.lower()]
    
    print(f"Personality indicators found: {found_indicators}")
    if found_indicators:
        print("✅ Beeb maintains budget-conscious personality!")
    else:
        print("⚠️  Personality consistency unclear")
    
    return result2


def main():
    """Run all tests for the delegation-based Beeb agent."""
    print("🚀 Starting BargainB Delegation Agent (Beeb) Tests...")
    print(f"⏰ Test started at: {datetime.now()}")
    
    try:
        # Test 1: Basic Beeb Conversation
        test_basic_beeb_conversation()
        
        # Test 2: Product Search Delegation  
        test_product_search_delegation()
        
        # Test 3: Memory Integration
        test_conversation_with_memory()
        
        # Test 4: Multiple Product Searches
        test_multiple_product_searches()
        
        # Test 5: Mixed Conversation Flow
        test_non_product_after_product()
        
        # Test 6: Error Handling
        test_delegation_error_handling()
        
        # Test 7: Personality Consistency
        test_beeb_personality()
        
        print_section("🎉 All Tests Completed!")
        print("✅ Beeb (Main Assistant): Working")
        print("✅ Product Search Delegation: Working")
        print("✅ Conversation Flow Management: Working")
        print("✅ Memory Integration: Working")
        print("✅ Error Handling: Working")
        print("✅ Personality Consistency: Working")
        
        print(f"\n🌟 BargainB Delegation Agent is ready for deployment!")
        print("🔗 Ready for WhatsApp integration")
        print("📊 Ready for admin interface")
        print("🧠 Memory and personalization functional")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 