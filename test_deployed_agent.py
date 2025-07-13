#!/usr/bin/env python3
"""
Test script for deployed Agent Beeb
Tests memory, product search, and conversation flow
"""

import requests
import json
import time
import uuid

# Configuration
API_URL = "https://agent-beeb-9dfc525b3a975fe7ade0fb8ffe654f53.us.langgraph.app"
API_KEY = "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"
HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

def test_thread_creation():
    """Test creating a new thread"""
    print("🧵 Testing thread creation...")
    
    response = requests.post(
        f"{API_URL}/threads",
        headers=HEADERS,
        json={}
    )
    
    if response.status_code == 200:
        thread_id = response.json()["thread_id"]
        print(f"✅ Thread created successfully: {thread_id}")
        return thread_id
    else:
        print(f"❌ Thread creation failed: {response.status_code} - {response.text}")
        return None

def wait_for_run_completion(thread_id, run_id, max_wait=60):
    """Wait for a run to complete and return the output"""
    print(f"⏳ Waiting for run {run_id} to complete...")
    
    start_time = time.time()
    while time.time() - start_time < max_wait:
        response = requests.get(
            f"{API_URL}/threads/{thread_id}/runs/{run_id}",
            headers=HEADERS
        )
        
        if response.status_code == 200:
            run_data = response.json()
            status = run_data.get("status")
            
            if status == "success":
                print(f"✅ Run completed successfully")
                return run_data.get("output", {})
            elif status == "error":
                print(f"❌ Run failed: {run_data.get('error', 'Unknown error')}")
                return None
            elif status in ["pending", "running"]:
                print(f"⏳ Run status: {status}")
                time.sleep(2)
            else:
                print(f"⚠️ Unknown status: {status}")
                time.sleep(2)
        else:
            print(f"❌ Failed to get run status: {response.status_code}")
            time.sleep(2)
    
    print(f"⏰ Run timed out after {max_wait} seconds")
    return None

def send_message(thread_id, message):
    """Send a message to the agent and get response"""
    print(f"💬 Sending: {message}")
    
    # Start the run
    response = requests.post(
        f"{API_URL}/threads/{thread_id}/runs",
        headers=HEADERS,
        json={
            "assistant_id": "memory_agent",
            "input": {"messages": [{"role": "user", "content": message}]}
        }
    )
    
    if response.status_code == 200:
        run_data = response.json()
        run_id = run_data["run_id"]
        
        # Wait for completion
        output = wait_for_run_completion(thread_id, run_id)
        
        if output and "messages" in output:
            assistant_message = output["messages"][-1]["content"]
            print(f"🤖 Agent Beeb: {assistant_message[:200]}...")
            return assistant_message
        else:
            print(f"⚠️ No output received from run")
            return None
    else:
        print(f"❌ Message failed: {response.status_code} - {response.text}")
        return None

def test_memory_functionality():
    """Test memory storage and recall"""
    print("\n🧠 Testing Memory Functionality...")
    
    # Create new thread
    thread_id = test_thread_creation()
    if not thread_id:
        return False
    
    # Set preferences
    print("\n📝 Setting user preferences...")
    response1 = send_message(thread_id, "Hi! I'm vegetarian, live in Amsterdam, on a tight budget, and prefer shopping at Albert Heijn.")
    
    if not response1:
        return False
    
    # Test memory recall
    print("\n🔍 Testing memory recall...")
    response2 = send_message(thread_id, "What do you remember about my preferences?")
    
    if not response2:
        return False
    
    # Check if preferences were remembered
    memory_keywords = ["vegetarian", "Amsterdam", "budget", "Albert Heijn"]
    remembered_count = sum(1 for keyword in memory_keywords if keyword.lower() in response2.lower())
    
    print(f"📊 Memory recall: {remembered_count}/{len(memory_keywords)} preferences remembered")
    
    return remembered_count >= 3

def test_product_search():
    """Test product search functionality"""
    print("\n🔍 Testing Product Search...")
    
    # Create new thread
    thread_id = test_thread_creation()
    if not thread_id:
        return False
    
    # Search for products
    print("\n🛒 Searching for vegetarian products...")
    response = send_message(thread_id, "Can you help me find some vegetarian protein options?")
    
    if not response:
        return False
    
    # Check if product search was triggered
    search_indicators = ["product", "price", "store", "vegetarian", "protein"]
    search_count = sum(1 for indicator in search_indicators if indicator.lower() in response.lower())
    
    print(f"📊 Product search: {search_count}/{len(search_indicators)} indicators found")
    
    return search_count >= 2

def test_cross_thread_memory():
    """Test memory persistence across threads"""
    print("\n🔄 Testing Cross-Thread Memory...")
    
    # First thread - set preferences
    thread1_id = test_thread_creation()
    if not thread1_id:
        return False
    
    print("\n📝 Setting preferences in first thread...")
    response1 = send_message(thread1_id, "I'm vegan and live in Utrecht, prefer organic products.")
    
    if not response1:
        return False
    
    # Second thread - check memory
    thread2_id = test_thread_creation()
    if not thread2_id:
        return False
    
    print("\n🔍 Checking memory in second thread...")
    response2 = send_message(thread2_id, "What do you know about my dietary preferences?")
    
    if not response2:
        return False
    
    # Check if preferences carried over
    memory_keywords = ["vegan", "Utrecht", "organic"]
    remembered_count = sum(1 for keyword in memory_keywords if keyword.lower() in response2.lower())
    
    print(f"📊 Cross-thread memory: {remembered_count}/{len(memory_keywords)} preferences remembered")
    
    return remembered_count >= 2

def test_conversation_flow():
    """Test natural conversation flow"""
    print("\n💬 Testing Conversation Flow...")
    
    # Create new thread
    thread_id = test_thread_creation()
    if not thread_id:
        return False
    
    # Multi-turn conversation
    messages = [
        "Hello! I need help with grocery shopping.",
        "I'm looking for healthy breakfast options.",
        "What would you recommend for someone on a budget?",
        "Can you help me find stores near me?"
    ]
    
    success_count = 0
    for message in messages:
        response = send_message(thread_id, message)
        if response and len(response) > 50:  # Meaningful response
            success_count += 1
        time.sleep(1)  # Small delay between messages
    
    print(f"📊 Conversation flow: {success_count}/{len(messages)} messages successful")
    
    return success_count >= 3

def main():
    """Run all tests"""
    print("🚀 Testing Agent Beeb Deployment")
    print("=" * 50)
    
    test_results = {
        "Memory Functionality": test_memory_functionality(),
        "Product Search": test_product_search(),
        "Cross-Thread Memory": test_cross_thread_memory(),
        "Conversation Flow": test_conversation_flow()
    }
    
    print("\n📊 TEST RESULTS")
    print("=" * 50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Agent Beeb is working correctly.")
    elif passed >= total * 0.7:
        print("⚠️ Most tests passed. Minor issues detected.")
    else:
        print("❌ Multiple test failures. Check deployment.")

if __name__ == "__main__":
    main() 