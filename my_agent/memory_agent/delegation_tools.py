"""
BargainB Delegation Tools

Tools for delegation between Beeb (main assistant) and specialized agents.
- ToProductSearch: Beeb delegates product queries to RAG agent
- CompleteOrEscalate: Specialized agents return results to Beeb
"""

from typing import Optional
from pydantic import BaseModel, Field


class ToProductSearch(BaseModel):
    """
    Delegates product search queries to the specialized product search RAG agent.
    
    This tool is used by Beeb when users ask about:
    - Finding specific products
    - Comparing prices
    - Getting product recommendations
    - Checking product availability
    """
    
    query: str = Field(
        description="The user's product search query or request"
    )
    dietary_preferences: Optional[str] = Field(
        description="Any dietary restrictions or preferences mentioned (vegetarian, gluten-free, etc.)",
        default=None
    )
    budget_preference: Optional[str] = Field(
        description="Budget considerations or price sensitivity (cheap, affordable, premium, etc.)",
        default=None
    )
    product_category: Optional[str] = Field(
        description="General product category if mentioned (dairy, produce, meat, etc.)",
        default=None
    )
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "query": "organic milk options",
                    "dietary_preferences": "organic",
                    "budget_preference": "affordable",
                    "product_category": "dairy"
                },
                {
                    "query": "cheap pasta for dinner tonight",
                    "dietary_preferences": None,
                    "budget_preference": "cheap",
                    "product_category": "pasta"
                },
                {
                    "query": "gluten-free bread options",
                    "dietary_preferences": "gluten-free",
                    "budget_preference": None,
                    "product_category": "bread"
                }
            ]
        }


class CompleteOrEscalate(BaseModel):
    """
    Tool for specialized agents to return results to Beeb or escalate if task cannot be completed.
    
    This tool is used by specialized agents (like product search) to:
    - Return search results to Beeb
    - Escalate if the query is outside their scope
    - Hand control back to Beeb for user response
    """
    
    cancel: bool = True  # Always True for specialized agents - they never respond directly
    reason: str = Field(
        description=(
            "Detailed information to pass back to Beeb. "
            "For product searches: Include product details, prices, store availability. "
            "For escalations: Explain why the task couldn't be completed."
        )
    )
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "cancel": True,
                    "reason": (
                        "Found 3 organic milk options: "
                        "1. Albert Heijn Organic Milk 1L - €1.89 at Albert Heijn "
                        "2. Campina Organic Milk 1L - €2.15 at Jumbo, €2.09 at Dirk "
                        "3. Melkunie Organic Milk 1L - €1.95 at PLUS "
                        "Best deal: Albert Heijn Organic Milk at €1.89"
                    )
                },
                {
                    "cancel": True,
                    "reason": (
                        "I successfully found several pasta options in your budget range. "
                        "The cheapest option is Barilla Spaghetti 500g for €0.89 at Dirk, "
                        "also available at Jumbo for €0.95. Would you like more details about "
                        "other pasta varieties or cooking suggestions?"
                    )
                },
                {
                    "cancel": True,
                    "reason": (
                        "I couldn't find any products matching that query. "
                        "Could you please clarify what specific type of product you're looking for? "
                        "I can help with groceries, household items, and food products."
                    )
                }
            ]
        } 