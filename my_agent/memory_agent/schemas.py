"""
BargainB Memory Agent Schemas

Defines the three types of long-term memory for personalizing grocery shopping:
1. UserProfile - User preferences, habits, and dietary information
2. ShoppingMemory - Past interactions, purchases, and decision outcomes  
3. Instructions - System behavior and personalization preferences

Based on mem.md patterns for reliable Trustcall integration.
"""

from typing import Optional, Literal, List
from datetime import datetime
from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    """User profile containing personal information and preferences for grocery shopping."""
    
    name: Optional[str] = Field(
        description="The user's name",
        default=None
    )
    
    location: Optional[str] = Field(
        description="The user's location (city, neighborhood)",
        default=None
    )
    
    dietary_restrictions: List[str] = Field(
        description="Dietary restrictions or preferences like vegetarian, gluten-free, etc.",
        default_factory=list
    )
    
    allergies: List[str] = Field(
        description="Food allergies the user has",
        default_factory=list
    )
    
    preferred_stores: List[str] = Field(
        description="User's preferred grocery stores",
        default_factory=list
    )
    
    budget_sensitivity: Optional[Literal["low", "medium", "high"]] = Field(
        description="How price-conscious the user is",
        default="medium"
    )
    
    family_size: Optional[int] = Field(
        description="Number of people in household",
        default=None
    )
    
    likes: List[str] = Field(
        description="Foods, brands, or product types the user enjoys",
        default_factory=list
    )
    
    dislikes: List[str] = Field(
        description="Foods, brands, or product types the user avoids",
        default_factory=list
    )


class ShoppingMemory(BaseModel):
    """Shopping memory storing past interactions and purchase decisions."""
    
    interaction_date: str = Field(
        description="Date of the interaction",
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d")
    )
    
    products_searched: List[str] = Field(
        description="Products the user searched for in this interaction",
        default_factory=list
    )
    
    products_purchased: List[str] = Field(
        description="Products the user indicated they purchased or planned to purchase",
        default_factory=list
    )
    
    price_sensitivity_shown: Optional[str] = Field(
        description="How price-sensitive the user was (low/medium/high)",
        default=None
    )
    
    feedback: Optional[str] = Field(
        description="User's feedback about recommendations or experience",
        default=None
    )
    
    outcome: Optional[str] = Field(
        description="Overall outcome of the interaction (helpful, not helpful, etc.)",
        default=None
    )


class Instructions(BaseModel):
    """Instructions for how the system should behave with this user."""
    
    content: str = Field(
        description="Instructions for how to personalize interactions with this user"
    )
    
    last_updated: str = Field(
        description="When these instructions were last updated",
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )


# Simple memory update schema for routing
class MemoryUpdate(BaseModel):
    """Decision on what memory type to update."""
    
    update_type: Literal['profile', 'shopping', 'instructions'] = Field(
        description="Type of memory to update"
    )
    
    context: str = Field(
        description="Context about what should be remembered or updated"
    ) 