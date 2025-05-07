from daily_shapes import household_items, clothing_and_accessories_items, school_university_items ,street_and_transportation_items, animals_and_plants, foods_and_drinks ,sports_and_entertainment , outdoor_nature_items
import random
import json

def generate_binary_relation_captions():
    """
    Generate binary relations between objects within the same category,
    without using color attributes.
    """
    relation_templates = [
        "a {obj1} is to the left of a {obj2}.",
        "a {obj1} is to the right of a {obj2}.",
        "a {obj1} is above a {obj2}.",
        "a {obj1} is below a {obj2}.",
        "a {obj1} is next to a {obj2}.",
        "a {obj1} is near a {obj2}.",
        "a {obj1} is partially behind a {obj2}."
        "a {obj1} is on a {obj2}.",
        "a {obj1} is under a {obj2}.",
        "a {obj1} is behind a {obj2}.",
        "a {obj1} is in front of a {obj2}.",
        "a {obj1} is between two {obj2}s."
    ]
    
    # Define category groups
    categories = {
        "household": household_items,
        "clothing": clothing_and_accessories_items,
        "school": school_university_items,
        "transportation": street_and_transportation_items,
        "animals_plants": animals_and_plants,
        "food_drinks": foods_and_drinks,
        "sports": sports_and_entertainment,
        "outdoors": outdoor_nature_items
    }
    
    
 
    
    captions = []
    
    # For each category
    for category_name, object_list in categories.items():
        print(f"Processing category: {category_name} with {len(object_list)} objects")
        
        # For each relation template
        for relation_template in relation_templates:
            # Skip pointing relation if not processing arrows
           
                
            # Generate combinations within this category
            for obj1,_ in object_list:
                # For pointing relation, only use arrow objects
                
                    
                for obj2,_ in object_list:
                    # Skip self-relations
                    
                        
                    caption = relation_template.format(obj1=obj1, obj2=obj2)
                    captions.append(caption)
    
    # Shuffle to ensure random order
    
    return captions

def generate_quantity_captions():
    """
    Generate captions describing different quantities of objects by category,
    without using color attributes.
    
    Returns:
        list: List of captions describing quantities of objects
    """
    # Configuration
    quantity_words = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    
    # Define category groups
    categories = {
        "household": household_items,
        "clothing": clothing_and_accessories_items,
        "school": school_university_items,
        "transportation": street_and_transportation_items,
        "animals_plants": animals_and_plants,
        "food_drinks": foods_and_drinks,
        "sports": sports_and_entertainment,
        "outdoors": outdoor_nature_items
    }
    
    # Generate all combinations
    captions = []
    
    for category_name, object_list in categories.items():
        for quantity_word in quantity_words:
            for obj,_ in object_list:
                # Format: "There are [quantity] [objects]."
                # Special case for "one"
                if quantity_word == "one":
                    # Use singular form: "There is one circle."
                    caption = f"There is {quantity_word} {obj}."
                else:
                    # Use plural form: "There are five squares."
                    # Simple pluralization rules
                    if obj.endswith('s') or obj.endswith('x') or obj.endswith('z'):
                        plural_obj = f"{obj}es"
                    elif obj.endswith('y') and not (obj.endswith('ay') or obj.endswith('ey') or obj.endswith('oy') or obj.endswith('uy')):
                        plural_obj = f"{obj[:-1]}ies"
                    else:
                        plural_obj = f"{obj}s"
                        
                    caption = f"There are {quantity_word} {plural_obj}."
                
                captions.append(caption)
    
    return captions
if __name__ == "__main__":
    # Generate captions
    binary_captions = generate_binary_relation_captions()
    quantity_captions = generate_quantity_captions()
    
    # Combine all captions
    all_captions = binary_captions + quantity_captions
    
    # Save to JSON file
    with open("daily_compositions_quantity.json", "w") as f:
        json.dump([{"caption": c} for c in quantity_captions], f, indent=2)
    with open("daily_compositions_relation.json", "w") as f:
        json.dump([{"caption": c} for c in binary_captions], f, indent=2)
    # Print statistics
    print(f"Generated {len(binary_captions)} binary relation captions")
    print(f"Generated {len(quantity_captions)} quantity captions")
    print(f"Total: {len(all_captions)} captions")
    
    # Print a few examples
    print("\nExamples of quantity captions:")
    for i, example in enumerate(random.sample(quantity_captions, min(3, len(quantity_captions)))):
        print(f"{i+1}. {example}")
    print("\nExamples of binary relation captions:")
    for i, example in enumerate(random.sample(binary_captions, min(3, len(binary_captions)))):
        print(f"{i+1}. {example}")