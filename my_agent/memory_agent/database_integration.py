"""
BargainB Memory Agent Database Integration

Integrates the memory agent with the BargainB product database
to provide personalized product recommendations based on user preferences.
"""

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

# Import the existing database utilities
from ..utils.database import BargainBDatabase


class MemoryAwareDatabase(BargainBDatabase):
    """
    Extended database class that provides memory-aware product searches
    and recommendations based on user preferences.
    """
    
    async def personalized_product_search(
        self, 
        query: str, 
        user_preferences: Optional[Dict[str, Any]] = None,
        budget_sensitivity: str = "medium",
        limit: int = 5
    ) -> List[Document]:
        """
        Search for products with personalization based on user memory.
        
        Args:
            query: Search query
            user_preferences: User semantic memory (likes, dislikes, dietary restrictions)
            budget_sensitivity: "low", "medium", or "high"
            limit: Maximum results
            
        Returns:
            Personalized product recommendations
        """
        
        # Get base product results
        base_results = await self.semantic_product_search(query, threshold=0.1, limit=limit*2)
        
        if not user_preferences:
            return base_results[:limit]
        
        # Filter and rank based on user preferences
        personalized_results = []
        
        for product in base_results:
            score = self._calculate_personalization_score(product, user_preferences)
            
            # Apply budget sensitivity
            if budget_sensitivity == "high":
                score = self._apply_budget_boost(product, score)
            elif budget_sensitivity == "low":
                score = self._apply_quality_boost(product, score)
            
            if score > 0:  # Only include relevant products
                personalized_results.append((product, score))
        
        # Sort by personalization score and return top results
        personalized_results.sort(key=lambda x: x[1], reverse=True)
        return [product for product, score in personalized_results[:limit]]
    
    
    def _calculate_personalization_score(self, product: Document, preferences: Dict[str, Any]) -> float:
        """
        Calculate how well a product matches user preferences.
        
        Args:
            product: Product document
            preferences: User preferences from semantic memory
            
        Returns:
            Personalization score (0-1, higher is better)
        """
        score = 0.5  # Base score
        content = product.page_content.lower()
        metadata = product.metadata
        
        # Check dietary restrictions (highest priority)
        dietary_restrictions = preferences.get("dietary_restrictions", [])
        for restriction in dietary_restrictions:
            restriction_lower = restriction.lower()
            
            if restriction_lower == "vegetarian":
                if any(term in content for term in ["vegetarian", "vegan", "plant-based"]):
                    score += 0.3
                elif any(term in content for term in ["meat", "chicken", "beef", "pork", "fish"]):
                    score -= 0.8  # Strong penalty
            
            elif restriction_lower == "vegan":
                if "vegan" in content or "plant-based" in content:
                    score += 0.3
                elif any(term in content for term in ["dairy", "milk", "cheese", "egg", "meat"]):
                    score -= 0.8
            
            elif restriction_lower == "gluten-free":
                if "gluten-free" in content or "glutenvrij" in content:
                    score += 0.3
                elif any(term in content for term in ["wheat", "gluten", "bread", "pasta"]):
                    score -= 0.6
        
        # Check allergies (critical)
        allergies = preferences.get("allergies", [])
        for allergy in allergies:
            allergy_lower = allergy.lower()
            if allergy_lower in content:
                score -= 1.0  # Strong penalty for allergens
        
        # Check likes (positive boost)
        likes = preferences.get("likes", [])
        for like in likes:
            if like.lower() in content:
                score += 0.2
        
        # Check dislikes (negative impact)
        dislikes = preferences.get("dislikes", [])
        for dislike in dislikes:
            if dislike.lower() in content:
                score -= 0.3
        
        # Preferred stores boost
        preferred_stores = preferences.get("preferred_stores", [])
        if preferred_stores and metadata.get("store_names"):
            store_names = metadata.get("store_names", "").lower()
            for store in preferred_stores:
                if store.lower() in store_names:
                    score += 0.1
        
        return max(0, min(1, score))  # Clamp between 0 and 1
    
    
    def _apply_budget_boost(self, product: Document, base_score: float) -> float:
        """Apply budget-conscious scoring boost."""
        metadata = product.metadata
        
        # Boost products with lower prices
        if metadata.get("best_price"):
            price = float(metadata["best_price"])
            if price < 2.0:
                base_score += 0.15
            elif price < 5.0:
                base_score += 0.1
        
        # Boost products on offer
        if metadata.get("has_offers"):
            base_score += 0.1
        
        return base_score
    
    
    def _apply_quality_boost(self, product: Document, base_score: float) -> float:
        """Apply quality-focused scoring boost."""
        content = product.page_content.lower()
        
        # Boost organic/premium products
        if any(term in content for term in ["organic", "bio", "premium", "artisan"]):
            base_score += 0.1
        
        return base_score
    
    
    async def get_category_recommendations(
        self, 
        category: str, 
        user_preferences: Optional[Dict[str, Any]] = None,
        limit: int = 10
    ) -> List[Document]:
        """
        Get category-based recommendations with personalization.
        
        Args:
            category: Product category
            user_preferences: User semantic memory
            limit: Maximum results
            
        Returns:
            Personalized category recommendations
        """
        
        # Get base category results
        base_results = await self.get_product_by_category(category, limit=limit*2)
        
        if not user_preferences:
            return base_results[:limit]
        
        # Apply personalization
        personalized_results = []
        
        for product in base_results:
            score = self._calculate_personalization_score(product, user_preferences)
            if score > 0.3:  # Minimum relevance threshold
                personalized_results.append((product, score))
        
        # Sort by score and return top results
        personalized_results.sort(key=lambda x: x[1], reverse=True)
        return [product for product, score in personalized_results[:limit]]
    
    
    def get_budget_recommendations(
        self, 
        max_budget: float,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Get budget-friendly product recommendations.
        
        Args:
            max_budget: Maximum budget per item
            user_preferences: User preferences
            
        Returns:
            List of budget recommendation strings
        """
        recommendations = []
        
        if not user_preferences:
            recommendations.append(f"Look for products under €{max_budget:.2f}")
            recommendations.append("Check store brand alternatives")
            recommendations.append("Look for items on offer")
            return recommendations
        
        # Dietary-specific budget tips
        dietary_restrictions = user_preferences.get("dietary_restrictions", [])
        
        if "vegetarian" in dietary_restrictions:
            recommendations.append("Legumes and beans are protein-rich and budget-friendly")
            recommendations.append("Check for store-brand vegetarian alternatives")
        
        if "vegan" in dietary_restrictions:
            recommendations.append("Plant-based proteins like lentils and chickpeas are affordable")
            recommendations.append("Look for seasonal vegetables for best prices")
        
        if "gluten-free" in dietary_restrictions:
            recommendations.append("Rice and potatoes are naturally gluten-free and cheap")
            recommendations.append("Check for gluten-free store brands")
        
        # General budget tips
        recommendations.append(f"Focus on products under €{max_budget:.2f}")
        recommendations.append("Compare prices across different stores")
        
        if user_preferences.get("preferred_stores"):
            stores = ", ".join(user_preferences["preferred_stores"])
            recommendations.append(f"Check weekly offers at {stores}")
        
        return recommendations


# Global instance for memory agent use
memory_db = MemoryAwareDatabase() 