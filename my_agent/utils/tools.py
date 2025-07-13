from langchain.tools import tool
from langchain_core.documents import Document
from typing import List
import asyncio

# Import BargainB database utility
from .database import db

class BargainBRetriever:
    """Custom retriever for BargainB database."""
    
    def __init__(self):
        self.db = db
    
    async def invoke(self, query: str) -> List[Document]:
        """
        Retrieve products from BargainB database using semantic search.
        
        Args:
            query: Search query
            
        Returns:
            List of Document objects with product information
        """
        return await self._async_invoke(query)
    
    async def _async_invoke(self, query: str) -> List[Document]:
        """Async version of invoke."""
        # Try semantic search first
        documents = await self.db.semantic_product_search(query, threshold=0.6, limit=10)
        
        # If no results, try category search
        if not documents:
            # Try to extract category from query
            category_keywords = {
                'milk': 'Zuivel en eieren',
                'bread': 'Brood en bakkerij',
                'fruit': 'Groenten en fruit',
                'meat': 'Vlees en vis',
                'cheese': 'Zuivel en eieren',
                'vegetables': 'Groenten en fruit',
                'snacks': 'Snoep en koekjes',
                'drinks': 'Dranken',
                'breakfast': 'Ontbijt',
                'dinner': 'Vlees en vis',
            }
            
            for keyword, category in category_keywords.items():
                if keyword in query.lower():
                    documents = await self.db.get_product_by_category(category, limit=10)
                    break
        
        return documents

# Global retriever instance
retriever = BargainBRetriever()

@tool
def retrieve_products(query: str) -> List[Document]:
    """Retrieve products from BargainB database using semantic search."""
    return retriever.invoke(query)

def format_documents(documents: List[Document]) -> str:
    """Format documents as a string for context."""
    return "\n\n".join([doc.page_content for doc in documents])

@tool
def format_documents_tool(documents: List[Document]) -> str:
    """Format documents as a string for context."""
    return "\n\n".join([doc.page_content for doc in documents])

@tool
def search_by_category(category: str, limit: int = 10) -> List[Document]:
    """Search products by category."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(db.get_product_by_category(category, limit))

@tool
def smart_grocery_search(query: str, budget: float = 100.0, store: str = None) -> List[Document]:
    """Smart grocery search with budget consideration."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(db.smart_grocery_search(query, budget, store))
