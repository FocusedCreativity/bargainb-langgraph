"""
Database connection utilities for BargainB product database on Supabase.
"""

import os
import asyncio
from typing import List, Dict, Any, Optional
import asyncpg
from langchain_core.documents import Document
from urllib.parse import urlparse
from datetime import datetime

class BargainBDatabase:
    """Database connection and query utility for BargainB on Supabase."""
    
    def __init__(self):
        self.connection = None
        
        # Get Supabase credentials from environment
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
        db_password = os.getenv('DB_PASSWORD')
        
        # Check if we have the required environment variables
        if not supabase_url or not supabase_key:
            print("⚠️  Database credentials not found in environment variables")
            print("   SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required")
            print("   Falling back to mock data mode")
            self.connection_params = None
            return
        
        # Parse Supabase URL to get connection parameters
        try:
            parsed_url = urlparse(supabase_url)
            
            # Extract project reference from URL (e.g., oumhprsxyxnocgbzosvh from https://oumhprsxyxnocgbzosvh.supabase.co)
            project_ref = parsed_url.hostname.split('.')[0]
            
            self.connection_params = {
                'host': f'aws-0-eu-west-3.pooler.supabase.com',  # Use session pooler for IPv4 compatibility
                'port': 5432,  # Session pooler port
                'database': 'postgres',
                'user': f'postgres.{project_ref}',  # Use project-specific user for pooler
                'password': db_password or 'AfdalX@20202',  # Use password from environment or fallback
                'ssl': 'prefer',  # Use prefer instead of require to avoid SSL issues
                'server_settings': {
                    'application_name': 'BargainB_Agent'
                }
            }
            
            print(f"🔗 Database configured for project: {project_ref}")
            
        except Exception as e:
            print(f"❌ Failed to parse database configuration: {e}")
            self.connection_params = None
    
    async def connect(self):
        """Establish database connection to Supabase."""
        if not self.connection_params:
            print("⚠️  Database connection not available - using mock data")
            return
            
        if not self.connection:
            try:
                # Add statement_cache_size=0 for pgbouncer compatibility
                conn_params = self.connection_params.copy()
                conn_params['statement_cache_size'] = 0
                
                self.connection = await asyncpg.connect(**conn_params)
                print("✅ Connected to BargainB database on Supabase")
            except Exception as e:
                print(f"❌ Supabase database connection failed: {e}")
                print("Connection parameters:")
                print(f"  Host: {self.connection_params['host']}")
                print(f"  Port: {self.connection_params['port']}")
                print(f"  Database: {self.connection_params['database']}")
                print(f"  User: {self.connection_params['user']}")
                print("  Falling back to mock data mode")
                self.connection_params = None  # Disable future connection attempts
                raise
    
    async def disconnect(self):
        """Close database connection."""
        if self.connection:
            await self.connection.close()
            self.connection = None
    
    async def semantic_product_search(self, query: str, threshold: float = 0.1, limit: int = 10) -> List[Document]:
        """
        Perform semantic search on products focused on finding best prices and deals.
        
        Args:
            query: Search query
            threshold: Similarity threshold
            limit: Maximum number of results
            
        Returns:
            List of Document objects with product information
        """
        if not self.connection:
            await self.connect()
        
        try:
            # Enhanced query to get individual store prices and identify best deals
            sql = """
            WITH search_results AS (
                SELECT 
                    product_id, gtin, title, brand, 
                    similarity_score, search_rank
                FROM semantic_product_search($1, $2, $3)
            ),
            store_pricing AS (
                SELECT 
                    sr.product_id,
                    sr.gtin,
                    sr.title,
                    sr.brand,
                    sr.similarity_score,
                    sr.search_rank,
                    s.name as store_name,
                    COALESCE(spr.promo_price, spr.price) as current_price,
                    spr.promo_price IS NOT NULL as on_offer,
                    ROW_NUMBER() OVER (PARTITION BY sr.product_id ORDER BY COALESCE(spr.promo_price, spr.price) ASC) as price_rank
                FROM search_results sr
                LEFT JOIN store_products sp ON sr.product_id = sp.product_id
                LEFT JOIN store_prices spr ON sp.id = spr.store_product_id
                LEFT JOIN stores s ON sp.store_id = s.id
                WHERE sp.is_available = true AND spr.price IS NOT NULL
            ),
            best_deals AS (
                SELECT 
                    product_id,
                    gtin,
                    title,
                    brand,
                    similarity_score,
                    search_rank,
                    MIN(current_price) as best_price,
                    JSON_AGG(
                        JSON_BUILD_OBJECT(
                            'store', store_name,
                            'price', current_price,
                            'on_offer', on_offer
                        ) ORDER BY current_price ASC
                    ) as store_prices
                FROM store_pricing
                GROUP BY product_id, gtin, title, brand, similarity_score, search_rank
            )
            SELECT 
                bd.*,
                p.description, p.quantity, p.unit
            FROM best_deals bd
            LEFT JOIN products p ON bd.product_id = p.id
            ORDER BY bd.search_rank;
            """
            
            rows = await self.connection.fetch(sql, query, threshold, limit)
            
            documents = []
            for row in rows:
                import json
                
                # Parse JSON safely
                store_prices = []
                if row['store_prices']:
                    try:
                        if isinstance(row['store_prices'], str):
                            store_prices = json.loads(row['store_prices'])
                        else:
                            store_prices = row['store_prices']
                    except (json.JSONDecodeError, TypeError):
                        store_prices = []
                
                if not store_prices:
                    continue  # Skip products without pricing
                
                # Find best price and identify deals
                best_store = None
                best_price = float('inf')
                offer_info = []
                
                for store_info in store_prices:
                    try:
                        price = float(store_info['price'])
                        store_name = store_info['store']
                        on_offer = store_info['on_offer']
                        
                        if price < best_price:
                            best_price = price
                            best_store = store_name
                        
                        if on_offer:
                            offer_info.append(f"{store_name} €{price:.2f} (ON OFFER)")
                        else:
                            offer_info.append(f"{store_name} €{price:.2f}")
                    except (KeyError, ValueError, TypeError):
                        continue  # Skip invalid price data
                
                if best_price == float('inf'):
                    continue  # Skip if no valid prices found
                
                # Check if query asks for description/details
                include_description = any(word in query.lower() for word in [
                    'describe', 'description', 'about', 'what is', 'ingredients', 
                    'nutrition', 'details', 'info', 'tell me about'
                ])
                
                # Create focused price-comparison content
                content_parts = [
                    f"Product: {row['title']}",
                    f"Brand: {row['brand'] or 'Unknown'}",
                    f"Size: {row['quantity'] or 'Unknown'}",
                    f"Best price: €{best_price:.2f} at {best_store}"
                ]
                
                # Add store pricing comparison
                content_parts.append(f"Stores: {', '.join(offer_info)}")
                
                # Only include description if specifically requested
                if include_description and row['description']:
                    content_parts.append(f"Description: {row['description']}")
                
                content = '\n'.join(content_parts)
                
                # Create metadata focused on pricing
                metadata = {
                    'id': str(row['product_id']),
                    'gtin': row['gtin'],
                    'title': row['title'],
                    'brand': row['brand'],
                    'similarity_score': float(row['similarity_score']),
                    'search_rank': float(row['search_rank']),
                    'best_price': best_price,
                    'best_store': best_store,
                    'store_prices': store_prices,
                    'has_offers': any(s.get('on_offer', False) for s in store_prices if isinstance(s, dict)),
                    'source': 'bargainb_database'
                }
                
                documents.append(Document(page_content=content, metadata=metadata))
            
            print(f"💰 Retrieved {len(documents)} products with pricing comparison")
            return documents
            
        except Exception as e:
            print(f"❌ Semantic search failed: {e}")
            return []
    
    async def get_product_by_category(self, category: str, limit: int = 10) -> List[Document]:
        """
        Get products by category using the built-in database function.
        
        Args:
            category: Category name
            limit: Maximum number of results
            
        Returns:
            List of Document objects with product information
        """
        if not self.connection:
            await self.connect()
        
        try:
            # Use the built-in category search function
            sql = """
            SELECT product_data FROM get_llm_products_by_category($1, $2);
            """
            
            rows = await self.connection.fetch(sql, category, limit)
            
            documents = []
            for row in rows:
                import json
                product_data = json.loads(row['product_data'])  # Parse JSON string
                
                # Extract data from the JSON object
                title = product_data.get('title', 'Unknown Product')
                brand = product_data.get('brand', 'Unknown')
                category_path = product_data.get('category_path', 'Unknown')
                ingredients = product_data.get('ingredients', [])
                nutrition_summary = product_data.get('nutrition_summary', 'Not specified')
                pricing = product_data.get('pricing', {})
                llm_search_text = product_data.get('llm_search_text', 'No description available')
                
                content = f"""
Product: {title}
Brand: {brand}
Category: {category_path}
Ingredients: {', '.join(ingredients) if ingredients else 'Not specified'}
Nutrition: {nutrition_summary}
Price Range: €{pricing.get('min_price', 'N/A')} - €{pricing.get('max_price', 'N/A')}
Available Stores: {pricing.get('available_stores', 0)} stores
Store Names: {', '.join(pricing.get('store_names', [])) if pricing.get('store_names') else 'Not specified'}

Product Description: {llm_search_text}
""".strip()
                
                metadata = {
                    'id': str(product_data.get('id', '')),
                    'title': title,
                    'brand': brand,
                    'category_path': category_path,
                    'ingredients': ingredients,
                    'nutrition_summary': nutrition_summary,
                    'pricing': pricing,
                    'llm_search_text': llm_search_text,
                    'source': 'bargainb_database'
                }
                
                documents.append(Document(page_content=content, metadata=metadata))
            
            print(f"📦 Retrieved {len(documents)} products from category: {category}")
            return documents
            
        except Exception as e:
            print(f"❌ Category search failed: {e}")
            return []
    
    async def smart_grocery_search(self, query: str, budget: float = 100.0, store: str = None) -> List[Document]:
        """
        Perform smart grocery search with budget consideration.
        
        Args:
            query: Search query
            budget: Budget limit
            store: Preferred store (optional)
            
        Returns:
            List of Document objects with product information
        """
        if not self.connection:
            await self.connect()
        
        try:
            # Use the built-in smart grocery search function
            sql = """
            SELECT 
                search_type, product_id, gtin, title, brand, 
                store_name, price, relevance_score, price_rank, suggestion
            FROM smart_grocery_search($1, $2, $3);
            """
            
            rows = await self.connection.fetch(sql, query, budget, store)
            
            documents = []
            for row in rows:
                # Get additional product details if needed
                product_details_sql = """
                SELECT 
                    p.description, p.quantity, p.unit,
                    c.level_1 || ' > ' || COALESCE(c.level_2, '') || ' > ' || COALESCE(c.level_3, '') || ' > ' || COALESCE(c.level_4, '') as category_path,
                    (SELECT string_agg(ingredient, ', ') FROM ingredients WHERE product_id = p.id) as ingredients,
                    (SELECT string_agg(feature, ', ') FROM features WHERE product_id = p.id) as features,
                    n.energy_kcal, n.proteins, n.carbohydrates, n.fat
                FROM products p
                LEFT JOIN categories c ON p.category_id = c.id
                LEFT JOIN nutrition n ON p.id = n.product_id
                WHERE p.id = $1
                """
                
                product_details = await self.connection.fetchrow(product_details_sql, row['product_id'])
                
                content = f"""
Product: {row['title']}
Brand: {row['brand'] or 'Unknown'}
Store: {row['store_name']}
Price: €{row['price']}
Price Rank: #{row['price_rank']} (cheapest option)
Suggestion: {row['suggestion']}
Relevance Score: {row['relevance_score']:.2f}
GTIN: {row['gtin'] or 'N/A'}
"""
                
                if product_details:
                    content += f"""
Category: {product_details['category_path'] or 'Unknown'}
Description: {product_details['description'] or 'No description available'}
Quantity: {product_details['quantity'] or 'Not specified'}
Unit: {product_details['unit'] or 'Not specified'}
Ingredients: {product_details['ingredients'] or 'Not specified'}
Features: {product_details['features'] or 'Not specified'}
Nutrition: Energy: {product_details['energy_kcal'] or 'N/A'} kcal, Protein: {product_details['proteins'] or 'N/A'}g, Carbs: {product_details['carbohydrates'] or 'N/A'}g, Fat: {product_details['fat'] or 'N/A'}g
"""
                
                content = content.strip()
                
                metadata = {
                    'id': str(row['product_id']),
                    'gtin': row['gtin'],
                    'title': row['title'],
                    'brand': row['brand'],
                    'store_name': row['store_name'],
                    'price': float(row['price']),
                    'price_rank': int(row['price_rank']),
                    'relevance_score': float(row['relevance_score']),
                    'suggestion': row['suggestion'],
                    'search_type': row['search_type'],
                    'source': 'bargainb_database'
                }
                
                if product_details:
                    metadata.update({
                        'category_path': product_details['category_path'],
                        'ingredients': product_details['ingredients'],
                        'features': product_details['features'],
                        'description': product_details['description'],
                        'quantity': product_details['quantity'],
                        'unit': product_details['unit'],
                    })
                
                documents.append(Document(page_content=content, metadata=metadata))
            
            print(f"💰 Retrieved {len(documents)} products within budget: €{budget}")
            return documents
            
        except Exception as e:
            print(f"❌ Smart grocery search failed: {e}")
            return []

# Global database instance
db = BargainBDatabase()


def semantic_search(query: str, limit: int = 10) -> List[dict]:
    """
    Synchronous wrapper for semantic product search.
    
    Args:
        query: Search query
        limit: Maximum number of results
        
    Returns:
        List of product dictionaries with pricing and details
    """
    import asyncio
    
    async def _search():
        db = BargainBDatabase()
        
        # Check if database is available
        if not db.connection_params:
            print("🔍 Using mock search data (database not available)")
            return _get_mock_search_results(query, limit)
            
        try:
            await db.connect()
            if not db.connection:
                print("🔍 Database connection failed, using mock data")
                return _get_mock_search_results(query, limit)
                
            documents = await db.semantic_product_search(query, limit=limit)
            # Convert Documents to dictionaries for easier use
            results = []
            for doc in documents:
                store_prices_json = _extract_store_prices_from_doc(doc)
                
                # Extract simple price from store_prices for compatibility
                import json
                import re
                try:
                    store_prices = json.loads(store_prices_json)
                    simple_price = store_prices[0]['price'] if store_prices else 'Price not available'
                except:
                    # Extract price from content as fallback
                    content = doc.page_content
                    price_match = re.search(r'Best price: €([\d.,]+)', content)
                    simple_price = f"€{price_match.group(1)}" if price_match else 'Price not available'
                
                result = {
                    'title': doc.metadata.get('title', 'Unknown Product'),
                    'brand': doc.metadata.get('brand', 'Unknown Brand'),
                    'quantity': doc.metadata.get('quantity', 'Unknown size'),
                    'price': simple_price,  # Simple price field for compatibility
                    'store_prices': store_prices_json,  # Detailed store prices JSON
                    'description': doc.metadata.get('description', ''),
                    'category': doc.metadata.get('category_path', 'Unknown'),
                    'gtin': doc.metadata.get('gtin', ''),
                    'content': doc.page_content
                }
                results.append(result)
            return results
        except Exception as e:
            print(f"🔍 Database search failed: {e}, using mock data")
            return _get_mock_search_results(query, limit)
        finally:
            if db.connection:
                await db.disconnect()
    
    # Simplified approach - just use asyncio.run
    try:
        return asyncio.run(_search())
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            # If we're in a running event loop, create a new thread
            import threading
            import queue
            
            result_queue = queue.Queue()
            exception_queue = queue.Queue()
            
            def run_search():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(_search())
                    result_queue.put(result)
                except Exception as e:
                    exception_queue.put(e)
                finally:
                    loop.close()
            
            thread = threading.Thread(target=run_search)
            thread.start()
            thread.join(timeout=30)  # 30 second timeout
            
            if not exception_queue.empty():
                print("🔍 Database search failed, using mock data")
                return _get_mock_search_results(query, limit)
            
            if not result_queue.empty():
                return result_queue.get()
            else:
                print("🔍 Database search timed out, using mock data")
                return _get_mock_search_results(query, limit)
        else:
            print("🔍 Database search failed, using mock data")
            return _get_mock_search_results(query, limit)


def _get_mock_search_results(query: str, limit: int = 10) -> List[dict]:
    """
    Provide mock search results when database is not available.
    
    Args:
        query: Search query
        limit: Maximum number of results
        
    Returns:
        List of mock product dictionaries
    """
    # Common grocery products with realistic Dutch pricing
    mock_products = [
        {
            'title': 'Organic Milk 1L',
            'brand': 'Biologisch',
            'quantity': '1L',
            'price': '€2.49',
            'store_prices': '[{"store": "Albert Heijn", "price": "€2.49", "on_offer": false}]',
            'description': 'Fresh organic whole milk',
            'category': 'Dairy',
            'gtin': '8712345678901',
            'content': 'Product: Organic Milk 1L\nBrand: Biologisch\nSize: 1L\nBest price: €2.49 at Albert Heijn'
        },
        {
            'title': 'Whole Wheat Bread',
            'brand': 'Hovis',
            'quantity': '800g',
            'price': '€1.89',
            'store_prices': '[{"store": "Jumbo", "price": "€1.89", "on_offer": false}]',
            'description': 'Nutritious whole wheat bread',
            'category': 'Bakery',
            'gtin': '8712345678902',
            'content': 'Product: Whole Wheat Bread\nBrand: Hovis\nSize: 800g\nBest price: €1.89 at Jumbo'
        },
        {
            'title': 'Free Range Eggs',
            'brand': 'Rondeel',
            'quantity': '12 pieces',
            'price': '€3.99',
            'store_prices': '[{"store": "Hoogvliet", "price": "€3.99", "on_offer": false}]',
            'description': 'Fresh free-range eggs',
            'category': 'Dairy',
            'gtin': '8712345678903',
            'content': 'Product: Free Range Eggs\nBrand: Rondeel\nSize: 12 pieces\nBest price: €3.99 at Hoogvliet'
        },
        {
            'title': 'Organic Bananas',
            'brand': 'Chiquita',
            'quantity': '1kg',
            'price': '€2.19',
            'store_prices': '[{"store": "Albert Heijn", "price": "€2.19", "on_offer": true}]',
            'description': 'Sweet organic bananas',
            'category': 'Fruits',
            'gtin': '8712345678904',
            'content': 'Product: Organic Bananas\nBrand: Chiquita\nSize: 1kg\nBest price: €2.19 at Albert Heijn (ON OFFER)'
        },
        {
            'title': 'Greek Yogurt',
            'brand': 'FAGE',
            'quantity': '500g',
            'price': '€2.99',
            'store_prices': '[{"store": "Jumbo", "price": "€2.99", "on_offer": false}]',
            'description': 'Creamy Greek yogurt',
            'category': 'Dairy',
            'gtin': '8712345678905',
            'content': 'Product: Greek Yogurt\nBrand: FAGE\nSize: 500g\nBest price: €2.99 at Jumbo'
        },
        {
            'title': 'Pasta Penne',
            'brand': 'Barilla',
            'quantity': '500g',
            'price': '€1.49',
            'store_prices': '[{"store": "Albert Heijn", "price": "€1.49", "on_offer": false}]',
            'description': 'Classic Italian pasta',
            'category': 'Pasta',
            'gtin': '8712345678906',
            'content': 'Product: Pasta Penne\nBrand: Barilla\nSize: 500g\nBest price: €1.49 at Albert Heijn'
        },
        {
            'title': 'Organic Tomatoes',
            'brand': 'Bio',
            'quantity': '500g',
            'price': '€2.79',
            'store_prices': '[{"store": "Jumbo", "price": "€2.79", "on_offer": false}]',
            'description': 'Fresh organic tomatoes',
            'category': 'Vegetables',
            'gtin': '8712345678907',
            'content': 'Product: Organic Tomatoes\nBrand: Bio\nSize: 500g\nBest price: €2.79 at Jumbo'
        },
        {
            'title': 'Chicken Breast',
            'brand': 'Scharrel',
            'quantity': '600g',
            'price': '€5.99',
            'store_prices': '[{"store": "Hoogvliet", "price": "€5.99", "on_offer": false}]',
            'description': 'Fresh chicken breast fillet',
            'category': 'Meat',
            'gtin': '8712345678908',
            'content': 'Product: Chicken Breast\nBrand: Scharrel\nSize: 600g\nBest price: €5.99 at Hoogvliet'
        }
    ]
    
    # Filter products based on query keywords
    query_lower = query.lower()
    relevant_products = []
    
    for product in mock_products:
        # Check if query matches product title, brand, description, or category
        if (query_lower in product['title'].lower() or 
            query_lower in product['brand'].lower() or
            query_lower in product['description'].lower() or
            query_lower in product['category'].lower() or
            any(keyword in product['title'].lower() for keyword in query_lower.split())):
            relevant_products.append(product)
    
    # If no specific matches, return a selection based on common food terms
    if not relevant_products:
        # Return a default selection
        relevant_products = mock_products[:limit]
    
    return relevant_products[:limit]


def _extract_store_prices_from_doc(doc: Document) -> str:
    """
    Extract store prices from a document and format as JSON string.
    
    Args:
        doc: Document with product information
        
    Returns:
        JSON string with store pricing information
    """
    import json
    import re
    
    # Extract pricing info from content field
    content = doc.page_content
    
    # Extract best price using regex
    best_price_match = re.search(r'Best price: €([\d.,]+) at ([^\\n]+)', content)
    if best_price_match:
        price = best_price_match.group(1)
        store = best_price_match.group(2)
        
        # Create price info structure  
        price_info = [{
            'store': store,
            'price': f"€{price}",
            'on_offer': False  # Default to false since we don't have promo price info
        }]
        
        return json.dumps(price_info)
    
    # Fallback to metadata if no content match
    price = doc.metadata.get('price', 0)
    store = doc.metadata.get('store_name', 'Unknown Store')
    
    # Create price info structure
    price_info = [{
        'store': store,
        'price': f"{price:.2f}",
        'on_offer': False  # Default to false since we don't have promo price info
    }]
    
    return json.dumps(price_info) 


def get_supabase_client():
    """Get a Supabase client for direct database access."""
    try:
        from supabase import create_client, Client
        
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
        
        if not supabase_url or not supabase_key:
            print("⚠️  Supabase credentials not found in environment variables")
            return None
            
        client: Client = create_client(supabase_url, supabase_key)
        return client
        
    except ImportError:
        print("⚠️  Supabase client not installed. Install with: pip install supabase")
        return None
    except Exception as e:
        print(f"⚠️  Failed to create Supabase client: {e}")
        return None

def log_message_truncation(user_id: str, thread_id: str, original_count: int, truncated_count: int, summary: str):
    """
    Log message truncation event to the Supabase database.
    
    Args:
        user_id: User identifier
        thread_id: Thread identifier
        original_count: Original message count
        truncated_count: Number of messages truncated
        summary: Generated summary
    """
    try:
        # Use the existing database connection pattern
        db = BargainBDatabase()
        
        # Run the logging in an async context
        async def _log_truncation():
            try:
                await db.connect()
                
                # Insert truncation log using asyncpg
                await db.connection.execute("""
                    INSERT INTO message_truncation_log (
                        user_id, thread_id, original_count, truncated_count, summary, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                """, user_id, thread_id, original_count, truncated_count, summary, datetime.now())
                
                await db.disconnect()
                
            except Exception as e:
                print(f"Error logging message truncation: {e}")
                await db.disconnect()
        
        # Run the async function
        asyncio.run(_log_truncation())
        
    except Exception as e:
        print(f"Error in log_message_truncation: {e}") 