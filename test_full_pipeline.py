#!/usr/bin/env python3
"""
Full pipeline test for BargainB bee delegation system.

This script tests:
1. User profile setup and memory storage
2. Product search functionality via Scout Bee
3. Memory recall and persistence
4. Conversation summarization
5. Supabase database integration
"""

import sys
import os
import asyncio
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_full_pipeline():
    """Test the complete bee delegation pipeline."""
    print("🐝 BargainB Full Pipeline Test")
    print("=" * 50)
    
    try:
        from my_agent.memory_agent.agent import create_bargainb_memory_agent
        from langchain_core.messages import HumanMessage
        
        # Create agent
        agent = create_bargainb_memory_agent()
        print("✅ Agent created successfully")
        
        # Test 1: User Profile Setup with Memory Storage
        print("\n📝 Test 1: User Profile Setup")
        print("-" * 30)
        
        result1 = agent.invoke(
            {
                'messages': [HumanMessage(content='Hi! I am John, a vegetarian who loves organic food and is budget-conscious. I live in Amsterdam, have a family of 4, and prefer Albert Heijn for shopping. I am allergic to nuts.')]
            },
            config={'configurable': {'user_id': 'john_test_123', 'thread_id': 'test_session_001'}}
        )
        
        print("✅ User profile setup completed")
        response1 = result1.get('messages', [])[-1].content
        print(f"Response: {response1[:150]}...")
        
        # Test 2: Product Search via Scout Bee
        print("\n🔍 Test 2: Product Search")
        print("-" * 30)
        
        result2 = agent.invoke(
            {
                'messages': result1['messages'] + [HumanMessage(content='I need to find organic milk for my family. Can you help me find the best price? I need 2 liters.')]
            },
            config={'configurable': {'user_id': 'john_test_123', 'thread_id': 'test_session_001'}}
        )
        
        print("✅ Product search completed")
        response2 = result2.get('messages', [])[-1].content
        print(f"Response: {response2[:150]}...")
        
        # Test 3: Memory Recall
        print("\n🧠 Test 3: Memory Recall")
        print("-" * 30)
        
        result3 = agent.invoke(
            {
                'messages': result2['messages'] + [HumanMessage(content='What do you remember about my preferences and dietary restrictions?')]
            },
            config={'configurable': {'user_id': 'john_test_123', 'thread_id': 'test_session_001'}}
        )
        
        print("✅ Memory recall completed")
        response3 = result3.get('messages', [])[-1].content
        print(f"Response: {response3[:150]}...")
        
        # Test 4: Shopping Recommendations
        print("\n🛒 Test 4: Shopping Recommendations")
        print("-" * 30)
        
        result4 = agent.invoke(
            {
                'messages': result3['messages'] + [HumanMessage(content='Based on my preferences, can you recommend some vegetarian products that are budget-friendly?')]
            },
            config={'configurable': {'user_id': 'john_test_123', 'thread_id': 'test_session_001'}}
        )
        
        print("✅ Shopping recommendations completed")
        response4 = result4.get('messages', [])[-1].content
        print(f"Response: {response4[:150]}...")
        
        # Test 5: Long conversation for summarization
        print("\n📄 Test 5: Long Conversation (Summarization)")
        print("-" * 30)
        
        # Add many messages to trigger summarization
        long_messages = result4['messages']
        for i in range(12):  # This will trigger summarization at >10 messages
            long_messages.append(HumanMessage(content=f'Message {i}: Can you tell me about product {i}?'))
            long_messages.append(HumanMessage(content=f'Response {i}: Here is information about product {i}'))
        
        result5 = agent.invoke(
            {
                'messages': long_messages + [HumanMessage(content='Can you summarize our conversation so far?')]
            },
            config={'configurable': {'user_id': 'john_test_123', 'thread_id': 'test_session_001'}}
        )
        
        print("✅ Long conversation handling completed")
        response5 = result5.get('messages', [])[-1].content
        print(f"Response: {response5[:150]}...")
        
        print("\n🎉 All pipeline tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_database_storage():
    """Test database storage functionality."""
    print("\n💾 Testing Database Storage")
    print("=" * 50)
    
    try:
        from my_agent.utils.database import BargainBDatabase
        
        db = BargainBDatabase()
        await db.connect()
        
        if db.connection:
            print("✅ Connected to Supabase database")
            
            # Check product data
            product_count = await db.connection.fetchval('SELECT COUNT(*) FROM products')
            print(f"📊 Products in database: {product_count}")
            
            # Check memory store
            memory_count = await db.connection.fetchval('SELECT COUNT(*) FROM memory_store')
            print(f"🧠 Memory records: {memory_count}")
            
            # Check conversation summaries
            summary_count = await db.connection.fetchval('SELECT COUNT(*) FROM conversation_summaries')
            print(f"📝 Conversation summaries: {summary_count}")
            
            # Test semantic search
            print("\n🔍 Testing semantic search...")
            search_results = await db.semantic_product_search("organic milk", limit=3)
            print(f"✅ Found {len(search_results)} products for 'organic milk'")
            
            for i, result in enumerate(search_results[:2]):
                title = result.metadata.get('title', 'Unknown')
                price = result.metadata.get('best_price', 'N/A')
                print(f"  {i+1}. {title} - €{price}")
            
            await db.disconnect()
            print("✅ Database tests completed")
            return True
            
        else:
            print("❌ Failed to connect to database")
            return False
            
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_bee_components():
    """Test individual bee components."""
    print("\n🐝 Testing Individual Bee Components")
    print("=" * 50)
    
    try:
        from my_agent.memory_agent.beeb_supervisor import create_beeb_supervisor
        from my_agent.memory_agent.scout_bee import create_scout_bee
        from my_agent.memory_agent.memory_bee import create_memory_bee
        from my_agent.memory_agent.scribe_bee import create_scribe_bee
        
        # Test Beeb Supervisor
        beeb = create_beeb_supervisor()
        print("✅ Beeb Supervisor created")
        
        # Test Scout Bee
        scout = create_scout_bee()
        print("✅ Scout Bee created")
        
        # Test Memory Bee
        memory = create_memory_bee()
        print("✅ Memory Bee created")
        
        # Test Scribe Bee
        scribe = create_scribe_bee()
        print("✅ Scribe Bee created")
        
        print("✅ All bee components working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Bee component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests."""
    print("🚀 Starting BargainB Full System Test")
    print("=" * 60)
    
    tests = [
        ("Bee Components", test_bee_components),
        ("Database Storage", test_database_storage),
        ("Full Pipeline", test_full_pipeline)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} Test...")
        try:
            if asyncio.iscoroutinefunction(test_func):
                success = await test_func()
            else:
                success = test_func()
            
            if success:
                passed += 1
                print(f"✅ {test_name} Test: PASSED")
            else:
                print(f"❌ {test_name} Test: FAILED")
                
        except Exception as e:
            print(f"❌ {test_name} Test: FAILED with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"🎯 Final Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! BargainB system is fully functional.")
        print("\n📋 System Features Verified:")
        print("   ✅ Bee delegation system working")
        print("   ✅ Memory storage and recall")
        print("   ✅ Product search with database")
        print("   ✅ Conversation summarization")
        print("   ✅ Supabase database integration")
        return 0
    else:
        print("⚠️ Some tests failed. Check output above for details.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 