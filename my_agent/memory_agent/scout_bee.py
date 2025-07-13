"""
Scout Bee 🐝🔍 - Product Search Specialist

Scout Bee is a specialized worker bee that handles all product searches,
price comparisons, and product recommendations for BargainB.

Scout Bee NEVER responds directly to users - it only searches for products
and returns results to Beeb (Queen Bee) who then responds to the user.
"""

from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from datetime import datetime

from my_agent.utils.database import semantic_search


@tool
def search_products(query: str, limit: int = 5) -> List[dict]:
    """
    Search for products using semantic similarity in BargainB database.
    
    Args:
        query: Product search query (e.g., "organic milk", "cheap pasta")
        limit: Maximum number of results to return
        
    Returns:
        List of product dictionaries with details and pricing
    """
    try:
        results = semantic_search(query, limit=limit)
        return results if results else [{"error": "No products found"}]
    except Exception as e:
        return [{"error": f"Search failed: {str(e)}"}]


@tool
def compare_prices(product_name: str, limit: int = 8) -> List[dict]:
    """
    Compare prices for a specific product across different stores.
    
    Args:
        product_name: Specific product to compare prices for
        limit: Maximum number of price comparisons to return
        
    Returns:
        List of price comparisons across stores
    """
    try:
        # Use semantic search to find the same product across stores
        results = semantic_search(product_name, limit=limit)
        
        if not results:
            return [{"error": f"No price data found for {product_name}"}]
        
        # Sort by price if available
        try:
            results.sort(key=lambda x: float(x.get('price', '999').replace('€', '').replace(',', '.')))
        except:
            pass  # If price sorting fails, return unsorted
            
        return results
    except Exception as e:
        return [{"error": f"Price comparison failed: {str(e)}"}]


@tool
def find_alternatives(product_query: str, category: str = None, limit: int = 5) -> List[dict]:
    """
    Find alternative products in the same category or with similar attributes.
    
    Args:
        product_query: Original product query
        category: Product category to search within (optional)
        limit: Maximum number of alternatives to return
        
    Returns:
        List of alternative product options
    """
    try:
        # Create broader search terms for alternatives
        search_terms = [
            product_query,
            f"{category} products" if category else "similar products",
            f"alternative to {product_query}"
        ]
        
        all_results = []
        for term in search_terms:
            results = semantic_search(term, limit=limit)
            all_results.extend(results)
        
        # Remove duplicates based on product name
        seen = set()
        unique_results = []
        for result in all_results:
            product_key = result.get('product_name', '')
            if product_key and product_key not in seen:
                seen.add(product_key)
                unique_results.append(result)
        
        return unique_results[:limit] if unique_results else [{"error": "No alternatives found"}]
    except Exception as e:
        return [{"error": f"Alternative search failed: {str(e)}"}]


def create_scout_bee():
    """
    Create Scout Bee, the specialized product search worker.
    
    Since the React agent is getting stuck, we'll use a simpler approach
    that directly calls the search tools and formats results.
    
    Returns:
        Simple function that processes search tasks
    """
    
    def scout_bee_processor(state):
        """
        Process search tasks directly without React agent complexity.
        
        Args:
            state: Task state with messages
            
        Returns:
            Updated state with search results
        """
        messages = state.get("messages", [])
        if not messages:
            return {
                "messages": [{
                    "role": "assistant",
                    "content": "No search task provided"
                }]
            }
        
        # Get the search query from the last message
        last_message = messages[-1]
        search_query = last_message.get("content", "") if isinstance(last_message, dict) else str(last_message)
        
        try:
            # Use search_products tool directly
            search_results = search_products.invoke({"query": search_query, "limit": 5})
            
            # Format results for Beeb
            if search_results and not (len(search_results) == 1 and search_results[0].get('error')):
                formatted_results = []
                for i, product in enumerate(search_results[:3]):  # Top 3 results
                    if product.get('error'):
                        continue
                    
                    title = product.get('title', 'Unknown Product')
                    brand = product.get('brand', 'Unknown Brand')
                    content = product.get('content', '')
                    
                    formatted_results.append(f"{i+1}. **{title}** by {brand}\n   {content}")
                
                if formatted_results:
                    response = f"🔍 Found {len(formatted_results)} products for '{search_query}':\n\n" + "\n\n".join(formatted_results)
                else:
                    response = f"I searched for '{search_query}' but couldn't find any suitable products."
            else:
                # Handle error or no results
                error_msg = search_results[0].get('error', 'No products found') if search_results else 'Search failed'
                response = f"I had trouble searching for '{search_query}': {error_msg}"
            
            return {
                "messages": messages + [{
                    "role": "assistant", 
                    "content": response
                }]
            }
            
        except Exception as e:
            error_response = f"Scout Bee encountered an error while searching for '{search_query}': {str(e)}"
            return {
                "messages": messages + [{
                    "role": "assistant",
                    "content": error_response
                }]
            }
    
    return scout_bee_processor


def format_search_results_for_beeb(results: List[dict], original_query: str) -> str:
    """
    Format Scout Bee's search results for Beeb to present to users.
    
    Args:
        results: List of product search results
        original_query: The original user query
        
    Returns:
        Formatted string with product information for Beeb
    """
    if not results:
        return f"I couldn't find any products matching '{original_query}'. Could you try a different search term?"
    
    # Check for errors
    if len(results) == 1 and results[0].get('error'):
        return f"I had trouble searching for '{original_query}': {results[0]['error']}"
    
    # Format successful results
    formatted_results = []
    best_deals = []
    
    for i, product in enumerate(results[:5]):  # Limit to top 5 results
        if product.get('error'):
            continue
            
        product_info = _format_single_product(product, i + 1)
        formatted_results.append(product_info)
        
        # Track best deals
        price_str = product.get('price', '')
        if price_str and '€' in price_str:
            try:
                price = float(price_str.replace('€', '').replace(',', '.'))
                store = product.get('store', 'Unknown store')
                best_deals.append((price, store, product.get('product_name', 'Unknown product')))
            except:
                pass
    
    if not formatted_results:
        return f"I found some results for '{original_query}' but couldn't process them properly. Please try a different search."
    
    # Build response
    response_parts = [
        f"Here's what I found for '{original_query}':\n"
    ]
    
    response_parts.extend(formatted_results)
    
    # Add best deal summary
    if best_deals:
        best_deals.sort(key=lambda x: x[0])  # Sort by price
        best_price, best_store, best_product = best_deals[0]
        response_parts.append(f"\n💰 **Best Deal**: {best_product} for €{best_price:.2f} at {best_store}")
    
    return "\n".join(response_parts)


def _format_single_product(product: dict, index: int) -> str:
    """
    Format a single product result for display.
    
    Args:
        product: Product dictionary with details
        index: Result number for display
        
    Returns:
        Formatted product information string
    """
    name = product.get('product_name', 'Unknown Product')
    price = product.get('price', 'Price not available')
    store = product.get('store', 'Store not specified')
    description = product.get('description', '')
    
    # Clean up price formatting
    if price and price != 'Price not available':
        if not price.startswith('€'):
            price = f"€{price}"
    
    # Build product info
    product_line = f"{index}. **{name}** - {price}"
    
    if store and store != 'Store not specified':
        product_line += f" at {store}"
    
    if description and len(description) > 0:
        # Truncate long descriptions
        if len(description) > 100:
            description = description[:97] + "..."
        product_line += f"\n   {description}"
    
    return product_line 