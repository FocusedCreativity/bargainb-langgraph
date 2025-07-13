"""
Product Search RAG Agent

Specialized agent that handles product searches using semantic RAG.
This agent NEVER responds directly to users - it only uses tools
and returns results to Beeb via CompleteOrEscalate.
"""

from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from datetime import datetime

from .delegation_tools import CompleteOrEscalate
from ..utils.database import semantic_search


# Convert existing semantic_search function to a proper tool
@tool 
def search_products(query: str, limit: int = 5) -> List[dict]:
    """
    Search for products using semantic similarity.
    
    Args:
        query: Product search query
        limit: Maximum number of results to return
        
    Returns:
        List of product dictionaries with details and pricing
    """
    try:
        results = semantic_search(query, limit=limit)
        return results
    except Exception as e:
        return [{"error": f"Search failed: {str(e)}"}]


def create_product_search_agent() -> Runnable:
    """
    Create the specialized product search RAG agent.
    
    This agent:
    - Receives delegated queries from Beeb
    - Uses semantic search to find products 
    - NEVER responds directly to users
    - Returns results to Beeb via CompleteOrEscalate
    
    Returns:
        Configured Runnable for product search agent
    """
    
    product_search_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a specialized product search agent for BargainB's grocery database.\n\n"
            
            "## CRITICAL: You NEVER respond directly to users ##\n"
            "- You are an internal agent that Beeb delegates to\n"
            "- Users don't know you exist - you work behind the scenes\n"
            "- ALWAYS use CompleteOrEscalate to return results to Beeb\n"
            "- NEVER generate conversational responses to users\n\n"
            
            "## Your Task ##\n"
            "When you receive a delegated query:\n"
            "1. Use search_products to find relevant items\n"
            "2. Analyze the results for best deals and options\n"
            "3. Format the information clearly for Beeb\n"
            "4. Use CompleteOrEscalate to return results\n\n"
            
            "## Search Strategy ##\n"
            "- Use the exact query first, then try variations if needed\n"
            "- Look for price comparisons across stores\n"
            "- Identify the best deals and value options\n"
            "- Include product details like size, brand, store availability\n\n"
            
            "## Information to Return to Beeb ##\n"
            "- Product names, brands, and sizes\n"
            "- Prices and which stores have the best deals\n"
            "- Availability across different store chains\n"
            "- Any special offers or promotions\n"
            "- Alternative suggestions if original query fails\n\n"
            
            "## Current Search Context ##\n"
            "User Query: {query}\n"
            "Dietary Preferences: {dietary_preferences}\n"
            "Budget Preference: {budget_preference}\n"
            "Product Category: {product_category}\n"
            "Time: {time}\n\n"
            
            "Remember: Use search_products then CompleteOrEscalate - never respond directly!"
        ),
        ("placeholder", "{messages}")
    ]).partial(time=datetime.now)
    
    # Available tools for product search agent
    product_search_tools = [search_products]
    
    # LLM for product search agent
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)  # Lower temp for more focused search
    
    # Product search agent with tools
    product_search_runnable = product_search_prompt | llm.bind_tools(
        product_search_tools + [CompleteOrEscalate]
    )
    
    return product_search_runnable


class ProductSearchAgent:
    """
    Wrapper class for product search agent to handle search delegation.
    """
    
    def __init__(self, runnable: Runnable):
        self.runnable = runnable
    
    def __call__(self, state, config: RunnableConfig):
        """
        Process delegated product search from Beeb.
        
        Args:
            state: Current conversation state
            config: Runtime configuration
            
        Returns:
            Updated state with search response (via CompleteOrEscalate)
        """
        while True:
            # Extract delegation context from the tool call
            delegation_context = self._extract_delegation_context(state)
            
            # Format state for product search context
            formatted_state = {
                **state,
                "query": delegation_context.get("query", ""),
                "dietary_preferences": delegation_context.get("dietary_preferences", "None"),
                "budget_preference": delegation_context.get("budget_preference", "None"),
                "product_category": delegation_context.get("product_category", "None")
            }
            
            # Generate search response
            result = self.runnable.invoke(formatted_state, config)
            
            # Ensure agent always uses tools or CompleteOrEscalate
            if not result.tool_calls:
                # Force escalation if no tools called
                messages = state["messages"] + [(
                    "user", 
                    "Please search for products using search_products tool, then use CompleteOrEscalate to return results."
                )]
                formatted_state = {**formatted_state, "messages": messages}
            else:
                break
        
        return {"messages": [result]}
    
    def _extract_delegation_context(self, state) -> dict:
        """
        Extract delegation context from Beeb's ToProductSearch tool call.
        
        Args:
            state: Current conversation state
            
        Returns:
            Dictionary with delegation parameters
        """
        # Look for the most recent ToProductSearch tool call
        for message in reversed(state["messages"]):
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    if tool_call.get("name") == "ToProductSearch":
                        return tool_call.get("args", {})
        
        # Fallback to empty context
        return {
            "query": "product search",
            "dietary_preferences": None,
            "budget_preference": None,
            "product_category": None
        }


def format_search_results_for_beeb(results: List[dict], original_query: str) -> str:
    """
    Format search results into a clear message for Beeb to use in user response.
    
    Args:
        results: List of product search results
        original_query: The original user query
        
    Returns:
        Formatted string with search results for Beeb
    """
    if not results or (len(results) == 1 and "error" in results[0]):
        return (
            f"I couldn't find any products matching '{original_query}'. "
            "Could you try being more specific about what you're looking for? "
            "I can help with groceries, household items, and food products available "
            "at Dutch supermarkets like Albert Heijn, Jumbo, and Dirk."
        )
    
    if len(results) == 1:
        product = results[0]
        return _format_single_product(product)
    
    # Multiple products - show comparison
    formatted_results = []
    best_deal = None
    lowest_price = float('inf')
    
    for i, product in enumerate(results[:5], 1):  # Top 5 results
        try:
            # Extract pricing info
            if 'store_prices' in product and product['store_prices']:
                prices = product['store_prices']
                if isinstance(prices, str):
                    import json
                    prices = json.loads(prices)
                
                # Find lowest price for this product
                product_prices = []
                for price_info in prices:
                    try:
                        price = float(price_info['price'])
                        store = price_info['store']
                        on_offer = price_info.get('on_offer', False)
                        
                        offer_text = " (ON OFFER)" if on_offer else ""
                        product_prices.append(f"{store} €{price:.2f}{offer_text}")
                        
                        # Track best overall deal
                        if price < lowest_price:
                            lowest_price = price
                            best_deal = f"{product.get('title', 'Unknown')} at {store} for €{price:.2f}"
                    except (ValueError, KeyError):
                        continue
                
                if product_prices:
                    formatted_results.append(
                        f"{i}. {product.get('title', 'Unknown')} "
                        f"({product.get('quantity', 'Unknown size')}) - "
                        f"{', '.join(product_prices)}"
                    )
            else:
                # No pricing info available
                formatted_results.append(
                    f"{i}. {product.get('title', 'Unknown')} "
                    f"({product.get('quantity', 'Unknown size')}) - Price info not available"
                )
        except Exception:
            formatted_results.append(f"{i}. {product.get('title', 'Product')} - Details unavailable")
    
    response = f"I found {len(results)} options for '{original_query}':\n\n"
    response += "\n".join(formatted_results)
    
    if best_deal:
        response += f"\n\n🏆 Best deal: {best_deal}"
    
    response += "\n\nWould you like more details about any of these options or help with something else?"
    
    return response


def _format_single_product(product: dict) -> str:
    """Format a single product result."""
    title = product.get('title', 'Unknown Product')
    quantity = product.get('quantity', 'Unknown size')
    
    if 'store_prices' in product and product['store_prices']:
        try:
            prices = product['store_prices']
            if isinstance(prices, str):
                import json
                prices = json.loads(prices)
            
            price_list = []
            best_price = float('inf')
            best_store = None
            
            for price_info in prices:
                try:
                    price = float(price_info['price'])
                    store = price_info['store']
                    on_offer = price_info.get('on_offer', False)
                    
                    if price < best_price:
                        best_price = price
                        best_store = store
                    
                    offer_text = " (ON OFFER)" if on_offer else ""
                    price_list.append(f"{store} €{price:.2f}{offer_text}")
                except (ValueError, KeyError):
                    continue
            
            if price_list:
                response = f"I found {title} ({quantity}):\n"
                response += f"Available at: {', '.join(price_list)}"
                if best_store:
                    response += f"\n\n💰 Best price: €{best_price:.2f} at {best_store}"
                return response
        except Exception:
            pass
    
    return f"I found {title} ({quantity}), but pricing information isn't available right now." 