from setuptools import setup, find_packages

setup(
    name="my_agent",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langgraph",
        "langchain-core",
        "langchain-community", 
        "langchain-openai",
        "langchain-text-splitters",
        "langchain",
        "beautifulsoup4",
        "lxml",
        "faiss-cpu",
        "tiktoken",
        "psycopg2-binary",
        "sqlalchemy",
        "asyncpg",
        "trustcall",
    ],
    python_requires=">=3.8",
) 