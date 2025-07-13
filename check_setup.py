#!/usr/bin/env python3
"""
Setup verification script for BargainB Self-RAG implementation.
Checks if all required dependencies and environment variables are properly configured.
"""

import os
import sys
import asyncio

def check_environment_variables():
    """Check if required environment variables are set."""
    print("🔍 Checking Environment Variables...")
    print("-" * 40)
    
    success = True
    
    # Check OpenAI API Key
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print("✅ OPENAI_API_KEY is set")
        print(f"   Key length: {len(openai_key)} characters")
    else:
        print("❌ OPENAI_API_KEY is not set")
        print("   Please set it using: export OPENAI_API_KEY='your-api-key-here'")
        success = False
    
    # Check Supabase URL
    supabase_url = os.getenv("SUPABASE_URL")
    if supabase_url:
        print("✅ SUPABASE_URL is set")
        print(f"   URL: {supabase_url}")
    else:
        print("❌ SUPABASE_URL is not set")
        print("   Please set it using: export SUPABASE_URL='your-supabase-url'")
        success = False
    
    # Check Supabase Service Role Key
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if supabase_key:
        print("✅ SUPABASE_SERVICE_ROLE_KEY is set")
        print(f"   Key length: {len(supabase_key)} characters")
    else:
        print("❌ SUPABASE_SERVICE_ROLE_KEY is not set")
        print("   Please set it using: export SUPABASE_SERVICE_ROLE_KEY='your-service-role-key'")
        success = False
    
    # Check Database Password
    db_password = os.getenv("DB_PASSWORD")
    if db_password:
        print("✅ DB_PASSWORD is set")
        print(f"   Password length: {len(db_password)} characters")
    else:
        print("⚠️  DB_PASSWORD is not set, using default from memory")
    
    # Check LangSmith API Key (optional)
    langsmith_key = os.getenv("LANGSMITH_API_KEY")
    if langsmith_key:
        print("✅ LANGSMITH_API_KEY is set (optional)")
        print(f"   Key length: {len(langsmith_key)} characters")
        print("   LangSmith tracing will be enabled")
    else:
        print("⚠️  LANGSMITH_API_KEY is not set (optional)")
        print("   LangSmith tracing will be disabled")
    
    return success

def check_dependencies():
    """Check if all required dependencies are installed."""
    print("\n🔍 Checking Dependencies...")
    print("-" * 40)
    
    required_packages = [
        "langgraph",
        "langchain_core",
        "langchain_community",
        "langchain_openai",
        "langchain_text_splitters",
        "langchain",
        "beautifulsoup4",
        "lxml",
        "faiss_cpu",
        "tiktoken",
        "pydantic",
        "asyncpg",
        "psycopg2",
        "sqlalchemy",
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            # Handle package name variations
            if package == "langchain_text_splitters":
                __import__("langchain_text_splitters")
            elif package == "langchain_core":
                __import__("langchain_core")
            elif package == "langchain_community":
                __import__("langchain_community")
            elif package == "langchain_openai":
                __import__("langchain_openai")
            elif package == "beautifulsoup4":
                __import__("bs4")
            elif package == "faiss_cpu":
                __import__("faiss")
            elif package == "psycopg2":
                __import__("psycopg2")
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("   Please install them using: pip install -r requirements.txt")
        return False
    
    return True

def test_openai_connection():
    """Test if OpenAI API key is valid."""
    print("\n🔍 Testing OpenAI Connection...")
    print("-" * 40)
    
    try:
        from langchain_openai import ChatOpenAI
        
        # Create a simple test
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        response = llm.invoke("Hello! This is a test.")
        
        print("✅ OpenAI connection successful")
        print(f"   Test response: {response.content[:50]}...")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI connection failed: {str(e)}")
        print("   Please check your API key and internet connection")
        return False

async def test_supabase_connection():
    """Test if Supabase database connection works."""
    print("\n🔍 Testing Supabase Database Connection...")
    print("-" * 40)
    
    try:
        from my_agent.utils.database import db
        
        # Test connection
        await db.connect()
        
        print("✅ Supabase connection successful")
        
        # Test a simple query
        if db.connection:
            result = await db.connection.fetch("SELECT 1 as test")
            print(f"   Test query successful: {result}")
        
        await db.disconnect()
        return True
        
    except Exception as e:
        print(f"❌ Supabase connection failed: {str(e)}")
        print("   Please check your SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY")
        return False

async def test_product_search():
    """Test product search functionality."""
    print("\n🔍 Testing Product Search...")
    print("-" * 40)
    
    try:
        from my_agent.utils.database import db
        
        # Test semantic search
        documents = await db.semantic_product_search("milk", threshold=0.6, limit=3)
        
        print(f"✅ Product search successful")
        print(f"   Found {len(documents)} products")
        
        if documents:
            print("   Sample products:")
            for i, doc in enumerate(documents[:2], 1):
                title = doc.metadata.get('title', 'Unknown Product')
                brand = doc.metadata.get('brand', 'Unknown Brand')
                print(f"   {i}. {title} - {brand}")
        
        return True
        
    except Exception as e:
        print(f"❌ Product search failed: {str(e)}")
        return False

def main():
    """Run all setup checks."""
    print("🚀 BargainB Self-RAG Setup Verification")
    print("=" * 50)
    
    # Run synchronous checks
    sync_checks = [
        check_environment_variables(),
        check_dependencies(),
        test_openai_connection(),
    ]
    
    # Run async checks
    async def run_async_checks():
        return [
            await test_supabase_connection(),
            await test_product_search(),
        ]
    
    try:
        async_checks = asyncio.run(run_async_checks())
    except Exception as e:
        print(f"❌ Async checks failed: {e}")
        async_checks = [False, False]
    
    all_checks = sync_checks + async_checks
    
    print("\n" + "=" * 50)
    print("📋 Setup Summary")
    print("=" * 50)
    
    if all(all_checks):
        print("✅ All checks passed! Your BargainB Self-RAG system is ready to use.")
        print("   You can now run: python test_self_rag.py")
    else:
        print("❌ Some checks failed. Please address the issues above.")
        print("   Refer to the README.md for detailed setup instructions.")
        sys.exit(1)

if __name__ == "__main__":
    main() 