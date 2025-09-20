# backend/product-service/app/utils/helpers.py
import re
from typing import Optional
from decimal import Decimal


def create_slug(text: str) -> str:
    """Create a URL-friendly slug from text"""
    # Convert to lowercase and replace spaces/special chars with hyphens
    slug = re.sub(r'[^\w\s-]', '', text.lower())
    slug = re.sub(r'[-\s]+', '-', slug)
    return slug.strip('-')


def format_price(price: Decimal) -> str:
    """Format price for display"""
    return f"${price:.2f}"


def calculate_discount_percentage(original_price: Decimal, sale_price: Decimal) -> Optional[float]:
    """Calculate discount percentage"""
    if original_price <= 0 or sale_price >= original_price:
        return None
    
    discount = original_price - sale_price
    percentage = (discount / original_price) * 100
    return round(percentage, 1)


def is_in_stock(product) -> bool:
    """Check if product is in stock"""
    if not product.track_inventory:
        return True
    
    if product.allow_backorder:
        return True
    
    return product.stock_quantity > 0


def validate_sku(sku: str) -> bool:
    """Validate SKU format"""
    if not sku:
        return False
    
    # SKU should be alphanumeric with optional hyphens/underscores
    pattern = r'^[A-Za-z0-9_-]+$'
    return bool(re.match(pattern, sku)) and 3 <= len(sku) <= 100


def parse_dimensions(dimensions: str) -> Optional[dict]:
    """Parse dimensions string into length, width, height"""
    if not dimensions:
        return None
    
    # Expected format: "L x W x H" (e.g., "10.5 x 8.2 x 3.0")
    pattern = r'^(\d+\.?\d*)\s*x\s*(\d+\.?\d*)\s*x\s*(\d+\.?\d*)$'
    match = re.match(pattern, dimensions.strip(), re.IGNORECASE)
    
    if match:
        return {
            "length": float(match.group(1)),
            "width": float(match.group(2)),
            "height": float(match.group(3))
        }
    
    return None


def generate_sku_suggestion(name: str, category: str = None) -> str:
    """Generate a SKU suggestion based on product name and category"""
    # Take first 3 letters of category (if provided)
    category_part = ""
    if category:
        category_part = re.sub(r'[^\w]', '', category)[:3].upper()
    
    # Take first letters of each word in product name
    name_words = re.findall(r'\w+', name.upper())
    name_part = ''.join(word[0] for word in name_words[:3])
    
    # Add a number for uniqueness (should be checked against existing SKUs)
    base_sku = f"{category_part}{name_part}"
    return base_sku if base_sku else "PROD"


class PriceCalculator:
    """Helper class for price calculations"""
    
    @staticmethod
    def add_tax(price: Decimal, tax_rate: Decimal) -> Decimal:
        """Add tax to price"""
        return price * (1 + tax_rate)
    
    @staticmethod
    def apply_discount(price: Decimal, discount_percent: float) -> Decimal:
        """Apply percentage discount to price"""
        if not (0 <= discount_percent <= 100):
            raise ValueError("Discount percentage must be between 0 and 100")
        
        discount_multiplier = Decimal(str(1 - (discount_percent / 100)))
        return price * discount_multiplier
    
    @staticmethod
    def calculate_margin(cost: Decimal, price: Decimal) -> float:
        """Calculate profit margin percentage"""
        if cost <= 0 or price <= 0:
            return 0.0
        
        margin = ((price - cost) / price) * 100
        return round(float(margin), 2)
