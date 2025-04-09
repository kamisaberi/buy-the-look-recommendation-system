import pandas as pd
import random

# Define attributes for product names
colors = ["blue", "red", "black", "white", "green", "yellow", "gray", "brown", "pink", "purple"]
materials = ["cotton", "wool", "leather", "polyester", "denim", "silk", "linen", "nylon"]
styles = ["slim-fit", "oversized", "striped", "plain", "checkered", "floral", "printed"]

# Define subcategories and base names
subcategories = {
    "tops": ["t-shirt", "polo shirt", "dress shirt", "blouse", "sweater", "hoodie", "jacket", "coat", "tank top"],
    "bottoms": ["jeans", "trousers", "shorts", "skirt", "leggings"],
    "footwear": ["sneakers", "boots", "sandals", "loafers", "heels", "flats"],
    "accessories": ["scarf", "hat", "belt", "gloves", "sunglasses"]
}

# Function to generate a product name
def generate_product_name(base_name):
    color = random.choice(colors)
    material = random.choice(materials) if random.random() < 0.5 else ""
    style = random.choice(styles) if random.random() < 0.5 else ""
    parts = [color, material, style, base_name]
    return " ".join(part for part in parts if part)

# Generate products
products = []
product_id = 1
for subcategory, base_names in subcategories.items():
    num_products = 250  # 250 per subcategory, totaling 1,000 products
    for base_name in base_names:
        count = num_products // len(base_names)
        for _ in range(count):
            name = generate_product_name(base_name)
            products.append({"product_id": product_id, "product_name": name, "subcategory": subcategory})
            product_id += 1
    # Distribute remaining products
    remaining = num_products % len(base_names)
    for i in range(remaining):
        base_name = base_names[i]
        name = generate_product_name(base_name)
        products.append({"product_id": product_id, "product_name": name, "subcategory": subcategory})
        product_id += 1

products_df = pd.DataFrame(products)

# Define look categories and preferences
look_categories = ["casual", "formal", "sporty", "business casual", "party"]
look_preferences = {
    "casual": {
        "tops": ["t-shirt", "hoodie", "sweater"],
        "bottoms": ["jeans", "shorts"],
        "footwear": ["sneakers", "sandals"]
    },
    "formal": {
        "tops": ["dress shirt", "blouse"],
        "bottoms": ["trousers", "skirt"],
        "footwear": ["loafers", "heels"]
    },
    "sporty": {
        "tops": ["t-shirt", "tank top"],
        "bottoms": ["shorts", "leggings"],
        "footwear": ["sneakers"]
    },
    "business casual": {
        "tops": ["polo shirt", "blouse"],
        "bottoms": ["trousers", "skirt"],
        "footwear": ["loafers", "flats"]
    },
    "party": {
        "tops": ["blouse", "jacket"],
        "bottoms": ["skirt", "trousers"],
        "footwear": ["heels", "boots"]
    }
}

# Generate looks
looks_list = []
look_id = 1
for _ in range(3500):  # 3,500 looks to ensure >10,000 rows
    category = random.choice(look_categories)
    look_products = []
    # Select top
    preferred_tops = look_preferences[category]["tops"]
    pattern = "|".join(preferred_tops)
    top_candidates = products_df[
        (products_df["subcategory"] == "tops") &
        (products_df["product_name"].str.contains(pattern, regex=True))
        ]
    if top_candidates.empty:
        top_candidates = products_df[products_df["subcategory"] == "tops"]
    if not top_candidates.empty:
        top_product = top_candidates.sample(1).iloc[0]
        look_products.append(top_product["product_id"])
    # Select bottom
    preferred_bottoms = look_preferences[category]["bottoms"]
    pattern = "|".join(preferred_bottoms)
    bottom_candidates = products_df[
        (products_df["subcategory"] == "bottoms") &
        (products_df["product_name"].str.contains(pattern, regex=True))
        ]
    if bottom_candidates.empty:
        bottom_candidates = products_df[products_df["subcategory"] == "bottoms"]
    if not bottom_candidates.empty:
        bottom_product = bottom_candidates.sample(1).iloc[0]
        look_products.append(bottom_product["product_id"])
    # Select footwear
    preferred_footwear = look_preferences[category]["footwear"]
    pattern = "|".join(preferred_footwear)
    footwear_candidates = products_df[
        (products_df["subcategory"] == "footwear") &
        (products_df["product_name"].str.contains(pattern, regex=True))
        ]
    if footwear_candidates.empty:
        footwear_candidates = products_df[products_df["subcategory"] == "footwear"]
    if not footwear_candidates.empty:
        footwear_product = footwear_candidates.sample(1).iloc[0]
        look_products.append(footwear_product["product_id"])
    # Optionally select accessory
    if random.random() < 0.5:
        accessory_candidates = products_df[products_df["subcategory"] == "accessories"]
        if not accessory_candidates.empty:
            accessory_product = accessory_candidates.sample(1).iloc[0]
            look_products.append(accessory_product["product_id"])
    # Add to looks_list
    for product_id in look_products:
        looks_list.append({"look_id": look_id, "category": category, "product_id": product_id})
    look_id += 1

looks_df = pd.DataFrame(looks_list)

# Save to CSV
products_df[["product_id", "product_name"]].to_csv("products.csv", index=False)
looks_df.to_csv("looks.csv", index=False)

print(f"Generated {len(products_df)} products and {len(looks_df)} look-product mappings.")
print("Data saved to 'products.csv' and 'looks.csv'.")