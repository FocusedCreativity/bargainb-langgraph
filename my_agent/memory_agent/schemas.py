"""
BargainB Memory Agent Schemas

Defines the three types of long-term memory for personalizing grocery shopping:
1. Semantic Memory - User preferences, habits, and dietary information
2. Episodic Memory - Past interactions, tool usage, and decision outcomes  
3. Procedural Memory - Instructions for system behavior and adaptation
"""

from typing import Optional, Literal, Dict, List, Any
from datetime import datetime
from pydantic import BaseModel, Field


class SemanticMemory(BaseModel):
    """
    Semantic memory stores facts about user preferences, habits, and emotional cues.
    
    Examples:
    - "likes": ["pasta", "avocado", "organic milk"]
    - "dislikes": ["spicy food", "seafood"]  
    - "budget_sensitivity": "high"
    - "dietary_restrictions": ["vegetarian", "gluten-free"]
    """
    
    # Food preferences
    likes: List[str] = Field(
        description="Foods, brands, or product types the user enjoys",
        default_factory=list,
        examples=[["pasta", "avocado", "organic milk", "Ben & Jerry's"]]
    )
    
    dislikes: List[str] = Field(
        description="Foods, brands, or product types the user avoids",
        default_factory=list,
        examples=[["spicy food", "seafood", "artificial sweeteners"]]
    )
    
    # Dietary information
    dietary_restrictions: List[str] = Field(
        description="Dietary restrictions or preferences",
        default_factory=list,
        examples=[["vegetarian", "gluten-free", "dairy-free", "kosher"]]
    )
    
    allergies: List[str] = Field(
        description="Food allergies the user has",
        default_factory=list,
        examples=[["peanuts", "shellfish", "eggs"]]
    )
    
    # Shopping behavior
    budget_sensitivity: Literal["low", "medium", "high"] = Field(
        description="How price-conscious the user is",
        default="medium"
    )
    
    preferred_stores: List[str] = Field(
        description="Stores the user prefers to shop at",
        default_factory=list,
        examples=[["Albert Heijn", "Jumbo", "PLUS"]]
    )
    
    # Lifestyle patterns
    household_size: Optional[int] = Field(
        description="Number of people in household",
        default=None
    )
    
    cooking_frequency: Literal["daily", "few_times_week", "weekly", "rarely"] = Field(
        description="How often the user cooks",
        default="few_times_week"
    )
    
    meal_types: List[str] = Field(
        description="Types of meals user typically makes",
        default_factory=list,
        examples=[["quick meals", "healthy meals", "family dinners", "meal prep"]]
    )


class EpisodicMemory(BaseModel):
    """
    Episodic memory stores specific past interactions, tool usage, and decision outcomes.
    
    Examples:
    - Tool usage: "2025-07-11: used meal planner → disliked tofu recipes"
    - Purchase decisions: "2025-07-10: bought organic milk at €2.18 from Jumbo"
    - Feedback: "2025-07-09: loved the pasta recipe suggestion"
    """
    
    date: datetime = Field(
        description="When this memory was created",
        default_factory=datetime.now
    )
    
    interaction_type: Literal[
        "product_search", "price_comparison", "meal_planning", 
        "recipe_suggestion", "shopping_list", "feedback", "purchase"
    ] = Field(
        description="Type of interaction that occurred"
    )
    
    user_action: str = Field(
        description="What the user did or requested",
        examples=[
            "searched for organic milk",
            "asked for quick dinner ideas", 
            "compared cheese prices",
            "gave feedback on recipe"
        ]
    )
    
    system_response: str = Field(
        description="How the system responded or what tools were used",
        examples=[
            "found 3 organic milk options, recommended Jumbo for best price",
            "suggested 5 quick pasta recipes",
            "compared prices across 4 stores"
        ]
    )
    
    user_feedback: Optional[str] = Field(
        description="User's reaction or feedback to the system response",
        default=None,
        examples=[
            "loved the recipe suggestion",
            "too expensive, wants cheaper options",
            "disliked tofu-based meals"
        ]
    )
    
    outcome: Literal["positive", "negative", "neutral", "unknown"] = Field(
        description="How successful the interaction was",
        default="unknown"
    )
    
    products_mentioned: List[str] = Field(
        description="Specific products that were discussed",
        default_factory=list,
        examples=[["Fairtrade Original Organic Coconut Milk", "Weetabix Original"]]
    )
    
    price_sensitivity_shown: Optional[Literal["high", "medium", "low"]] = Field(
        description="Level of price sensitivity shown in this interaction",
        default=None
    )


class ProceduralMemory(BaseModel):
    """
    Procedural memory stores instructions for system behavior, tone, and adaptation.
    
    Examples:
    - "Be budget-aware and always show cheapest options first"
    - "User prefers friendly tone and quick meal suggestions"
    - "Always ask about dietary restrictions before suggesting recipes"
    """
    
    communication_style: str = Field(
        description="How the system should communicate with this user",
        default="friendly and helpful",
        examples=[
            "friendly and helpful",
            "concise and direct", 
            "detailed and educational",
            "casual and humorous"
        ]
    )
    
    budget_behavior: str = Field(
        description="How to handle pricing and budget considerations",
        default="show best value options",
        examples=[
            "always prioritize cheapest options",
            "show best value options", 
            "focus on quality over price",
            "provide price ranges for planning"
        ]
    )
    
    recommendation_style: str = Field(
        description="How to make product and meal recommendations",
        default="balanced suggestions with options",
        examples=[
            "quick and easy meal focus",
            "healthy options priority",
            "family-friendly suggestions",
            "balanced suggestions with options"
        ]
    )
    
    personalization_level: Literal["low", "medium", "high"] = Field(
        description="How much to personalize responses based on past interactions",
        default="medium"
    )
    
    proactive_suggestions: bool = Field(
        description="Whether to offer unsolicited helpful suggestions",
        default=True
    )
    
    custom_instructions: List[str] = Field(
        description="Specific instructions or rules for this user",
        default_factory=list,
        examples=[[
            "Always mention store promotions when available",
            "Suggest seasonal ingredients",
            "Remind about dietary restrictions"
        ]]
    )
    
    updated_at: datetime = Field(
        description="When these instructions were last updated",
        default_factory=datetime.now
    )


class MemoryUpdate(BaseModel):
    """Decision on what memory type to update"""
    update_type: Literal['semantic', 'episodic', 'procedural'] = Field(
        description="Which type of memory to update based on the user interaction"
    ) 