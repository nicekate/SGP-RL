
color_list = ["red", "blue", "green", "yellow", "black", "white", "orange", "purple", "pink", "gray"]
# Define a list of tuples (shape name, description)
basic_shapes = [
    ("point", "a single point."),
    ("line", "a straight segment between two points."),
    ("ray", "a line extending infinitely in one direction."),
    ("line segment", "a finite straight line between two endpoints."),
    ("polyline", "a connected series of line segments."),
    
    ("circle", "all points on it are equidistant from a center."),
    ("ellipse", "an elongated circle."),
    ("semi-circle", "half of a circle."),
    ("quarter circle", "one-fourth arc of a circle."),
    ("annulus", "a ring shape formed by two concentric circles."),

    ("equilateral triangle", "a triangle with all sides and angles equal."),
    ("isosceles triangle", "a triangle with two equal sides."),
    ("scalene triangle", "a triangle with all sides of different lengths."),
    ("right triangle", "a triangle with one 90-degree angle."),

    ("square", "a quadrilateral with equal sides and 90-degree angles."),
    ("rectangle", "a quadrilateral with opposite sides equal and right angles."),
    ("rhombus", "a quadrilateral with all sides equal but not necessarily right angles."),
    ("parallelogram", "a quadrilateral with opposite sides parallel."),
    ("trapezoid", "a quadrilateral with at least one pair of parallel sides."),
    ("kite", "a quadrilateral with two pairs of adjacent sides equal."),

    ("pentagon", "a polygon with 5 sides."),
    ("hexagon", "a polygon with 6 sides."),
    ("heptagon", "a polygon with 7 sides."),
    ("octagon", "a polygon with 8 sides."),
    ("nonagon", "a polygon with 9 sides."),
    ("decagon", "a polygon with 10 sides."),
    ("dodecagon", "a polygon with 12 sides."),
    

    ("arc", "a section of a circle."),
    ("sector", "a pie-slice shape bounded by two radii and an arc."),    
]


# List of regular polygons with names and short descriptions
regular_polygons = [
    ("equilateral triangle", "a triangle with all three sides and angles equal."),
    ("square", "a four-sided polygon with equal sides and 90-degree angles."),
    ("regular pentagon", "a five-sided polygon with equal sides and angles."),
    ("regular hexagon", "a six-sided polygon with equal sides and angles."),
    ("regular heptagon", "a seven-sided polygon with equal sides and angles."),
    ("regular octagon", "an eight-sided polygon with equal sides and angles."),
    ("regular nonagon", "a nine-sided polygon with equal sides and angles."),
    ("regular decagon", "a ten-sided polygon with equal sides and angles."),
    ("regular hendecagon", "an eleven-sided polygon with equal sides and angles."),
    ("regular dodecagon", "a twelve-sided polygon with equal sides and angles."),
    ("regular tridecagon", "a thirteen-sided polygon with equal sides and angles."),
    ("regular tetradecagon", "a fourteen-sided polygon with equal sides and angles."),
    ("regular pentadecagon", "a fifteen-sided polygon with equal sides and angles."),
    ("regular hexadecagon", "a sixteen-sided polygon with equal sides and angles."),
    ("regular heptadecagon", "a seventeen-sided polygon with equal sides and angles."),
    ("regular octadecagon", "an eighteen-sided polygon with equal sides and angles."),
    ("regular enneadecagon", "a nineteen-sided polygon with equal sides and angles."),
    ("regular icosagon", "a twenty-sided polygon with equal sides and angles.")
]


composite_shapes = [
    # Stars and Multi-arm Shapes
    ("5-pointed star", "a star with five evenly spaced points."),
    ("6-pointed star", "a star with six points, often known as the star of david."),
    ("7-pointed star", "a star with seven sharp points."),
    ("8-pointed star", "a star with eight radiating points."),
    

    # Arrows and Directional Shapes
    ("right arrow", "an arrow pointing to the right."),
    ("left arrow", "an arrow pointing to the left."),
    ("up arrow", "an arrow pointing upward."),
    ("down arrow", "an arrow pointing downward."),
    ("bidirectional arrow", "an arrow with two heads pointing in opposite directions."),
    ("curved arrow", "an arrow that follows a curved path."),
    ("circular arrow", "an arrow that loops around on a circle."),
    
    # Symbols and Iconic Shapes
    ("heart", "a symmetrical shape representing love."),
    ("crescent", "a curved shape like a partial moon."),
    ("cross", "two intersecting lines forming four right angles."),
    ("plus sign", "a horizontal line and a vertical line crossing at the center."),
    ("minus sign", "a single horizontal bar."),
    ("infinity symbol", "a sideways figure-eight representing infinity."),
    ("lightning bolt", "a jagged, angular shape symbolizing electricity."),
    ("target symbol", "a series of concentric circles resembling a bullseye."),
    ("recycling symbol", "three chasing arrows forming a triangle to represent sustainability."),

    
    # Miscellaneous Composite Shapes
    ("spiral", "a shape that winds outward from a central point."),
    ("wave shape", "a smooth, sinusoidal curve resembling a wave."),
    ("zigzag line", "a line alternating sharp angles back and forth."),
    ("cloud shape", "a puffy outline resembling a cartoon cloud."),
    ("speech bubble", "a rounded shape used to contain spoken text."),
    ("thought bubble", "a cloud-like shape used to show inner thoughts."),
    ("gear shape", "a circular shape with evenly spaced teeth around the edge."),
    ("keyhole", "a shape used for locking mechanisms, often circular with a notch."),
    ("shield", "a protective shape often used in heraldry."), 
    ("badge", "a shape used to highlight or label content."),
    ("label tag", "a tag-like shape often used to identify items."),
    ("pie chart slice", "a sector from a circle representing a portion of it."),
    ("venn diagram", "overlapping circles showing logical relationships."),

    # Shapes from Overlapping Geometry
    ("lens", "an oval shape formed by the overlap of two circles."),
    ("droplet", "a rounded teardrop-like shape."),
    ("clover shape", "a compound shape made from overlapping circular lobes."),
    ("petal shape", "an almond-like shape resembling a flower petal.")
]


object_names = [name for name, _ in basic_shapes + regular_polygons + composite_shapes]

# Generate lowercase captions
captions = [f"This is a simple geometric object, a {color} {name}, {desc}" for name, desc in basic_shapes + regular_polygons + composite_shapes for color in color_list]

if __name__ == "__main__":
    # Print the generated captions
    import json
    json_path = "simple_shapes.json"
    with open(json_path, 'w') as f:
        json.dump([{"caption": c} for c in captions], f, indent=2)
    
    # Print the number of lines
    print(f"Saved {len(captions)} captions to {json_path}")
    
    # Optional: Print a few example captions
    print("\nExample captions:")
    for i in range(min(3, len(captions))):
        print(f"{i+1}. {captions[i]}")



