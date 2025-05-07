from simple_shapes import object_names

import random
import json
color_list = ["red", "blue", "green", "yellow", "black", "white", "orange", "purple", "pink", "gray"]
def generate_binary_relation_captions(num_captions = 9000):
    """
    Generate all possible combinations of objects and relations with colors,
    ensuring that for 'pointing at' relations, obj1 must have 'arrow' in its name.
    """
    
    relation_templates = [
        "a {color1} {obj1} is to the left of a {color2} {obj2}.",
        "a {color1} {obj1} is to the right of a {color2} {obj2}.",
        "a {color1} {obj1} is above a {color2} {obj2}.",
        "a {color1} {obj1} is below a {color2} {obj2}.",
        "a {color1} {obj1} is next to a {color2} {obj2}.",
        "a {color1} {obj1} is near a {color2} {obj2}.",
        "a {color1} {obj1} overlaps with a {color2} {obj2}.",
        "a {color1} {obj1} is pointing at a {color2} {obj2}.",
        "a {color1} {obj1} is inside a {color2} {obj2}.",
        "a {color1} {obj1} is partially behind a {color2} {obj2}."
    ]
    
    # Filter arrow objects for the pointing relation
    arrow_objects = [obj for obj in object_names if 'arrow' in obj]
    
    captions = []
    for relation_template in relation_templates:
        # For pointing relation, only use arrow objects as obj1
        if "pointing at" in relation_template:
            obj1_list = arrow_objects
        else:
            obj1_list = object_names
        
        # Generate a reasonable number of color combinations
        for color1 in color_list:
            for color2 in color_list:
                # Generate all combinations
                for obj1 in obj1_list:
                    for obj2 in object_names:
                        caption = relation_template.format(
                            color1=color1, 
                            obj1=obj1, 
                            color2=color2, 
                            obj2=obj2
                        )
                        captions.append(caption)
    
    return captions



def generate_quantity_captions():
    """
    Generate captions describing different quantities of colored objects.
    Creates all combinations of color, object, and quantity.
    
    Returns:
        list: List of captions describing quantities of objects
    """
    # Configuration
    quantity_words = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    
    # Generate all combinations
    captions = []
    
    for quantity_word in quantity_words:
        for color in color_list:
            for obj in object_names:
                # Format: "There are [quantity] [color] [objects]."
                # Special case for "one"
                if quantity_word == "one":
                    # Use singular form: "There is one red circle."
                    caption = f"There is {quantity_word} {color} {obj}."
                else:
                    # Use plural form: "There are five blue squares."
                    # Need to handle pluralization of object names
                    plural_obj = obj
                    # Simple pluralization rules - could be enhanced for special cases
                    if obj.endswith('s') or obj.endswith('x') or obj.endswith('z'):
                        plural_obj = f"{obj}es"
                    elif obj.endswith('y') and not (obj.endswith('ay') or obj.endswith('ey') or obj.endswith('oy') or obj.endswith('uy')):
                        plural_obj = f"{obj[:-1]}ies"
                    else:
                        plural_obj = f"{obj}s"
                        
                    caption = f"There are {quantity_word} {color} {plural_obj}."
                
                captions.append(caption)
    
    return captions

if __name__ == "__main__":
    # Generate captions
    binary_captions = generate_binary_relation_captions()
    quantity_captions = generate_quantity_captions()
    
    # Combine all captions
    all_captions = binary_captions + quantity_captions
    
    # Save to JSON file
    with open("simple_compositions_quantity.json", "w") as f:
        json.dump([{"caption": c} for c in quantity_captions], f, indent=2)
    with open("simple_compositions_relation.json", "w") as f:
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