import cairosvg
from io import BytesIO
from func_timeout import func_timeout, FunctionTimedOut, func_set_timeout
from lxml import etree
import os
from PIL import Image
import re
def get_svg_code_length(svg_code):
    """
    Calculate the length of SVG code excluding whitespace characters and comments.
    
    Args:
        svg_code (str): The SVG content as a string
        
    Returns:
        int: Length of the SVG code without whitespace and comments
    """
    if not svg_code:
        return 0
    
    # Remove XML comments <!-- ... -->
    no_xml_comments = re.sub(r'<!--.*?-->', '', svg_code, flags=re.DOTALL)
    
    # Remove CSS comments /* ... */ (which might appear in <style> tags)
    no_comments = re.sub(r'/\*.*?\*/', '', no_xml_comments, flags=re.DOTALL)
    
    # Remove all whitespace characters (space, tab, newline)
    cleaned_svg = re.sub(r'\s+', '', no_comments)
    
    return len(cleaned_svg)

def safe_svg_to_image(svg_code, timeout=0.1):
    """Convert SVG to image with timeout protection."""
    # Get process info for debugging
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    global_rank = int(os.environ.get("RANK", "0"))
    
    
    # Skip empty SVG
    if not svg_code :
        # print(f"[Rank {global_rank}/{local_rank}] Empty SVG")
        return None
    
    try:
        
        result = svg_to_image(svg_code)
        return result
    except FunctionTimedOut:
        
        # print(f"[Rank {global_rank}/{local_rank}] SVG to image conversion timed out")
        # print(f"The SVG code is: {svg_code}")
        return None
    except Exception as e:
        # print(f"[Rank {global_rank}/{local_rank}] Error in svg_to_image: {type(e).__name__}: {str(e)}")
        return None
@func_set_timeout(3)
def svg_to_image(svg_code):
    """
    Attempts to parse and recover from errors in SVG code,
    then renders the recovered SVG to a PNG image.

    Parameters:
      svg_code (str): The SVG content as a string.
      output_filename (str): The filename for the output image.
    """
    # Create an XML parser in recovery mode. This tells lxml
    # to try to recover as much as possible from broken XML.
    
    try:
        parser = etree.XMLParser(recover=True)
        
        
        
        tree = etree.fromstring(svg_code.encode('utf-8'), parser)
        valid_svg = etree.tostring(tree)
        
        
        
        png_data = cairosvg.svg2png(bytestring=valid_svg)
        # png_data = cairosvg.svg2png(bytestring=svg_code.encode('utf-8'))
        image = Image.open(BytesIO(png_data))
        return image
    except Exception as e:
        # black_image = Image.new('RGB', (256, 256), color='black')
        return None

def extract_svg(text):
    if text is None:
        return ""
    # ans = SVGReward.extract_answer(text)
    complete_matches = re.findall(r'(<svg .*?</svg>)', text, re.DOTALL)
    if complete_matches:
        return complete_matches[-1].strip()  # Return the last match
    
    # If no complete SVG found, find the last opening SVG tag
    start_idx = text.rfind("<svg")
    if start_idx >= 0:
        # Extract from the last <svg tag to the end of the text
        return text[start_idx:].strip()
        
    return ""




from pathlib import Path
import re
import xml.etree.ElementTree as ET


_FILL_RE = re.compile(r"fill\s*:\s*([^;]+)", re.I)
def is_sketch_style(svg_source: str) -> bool:
    """
    Return True if every paintable element in the SVG is explicitly unfilled
    (i.e. fill='none' or fill='transparent'). Elements without fill specified
    are considered filled (since SVG default is black fill).
    
    Args:
        svg_source (str): Raw SVG content as a string
        
    Returns:
        bool: True if the SVG is in sketch style (no fills), False otherwise
    """
    # Parse directly from the SVG string
    try:
        root = ET.fromstring(svg_source)
    except ET.ParseError:
        # Return False for invalid SVG
        return False

    # Helper: does this element have a fill? (default is filled)
    def filled(elem: ET.Element) -> bool:
        # Explicit attribute check
        if "fill" in elem.attrib:
            fill_attr = elem.attrib.get("fill").lower()
            return fill_attr not in ("none", "transparent")
        
        # CSS in style="..." check
        style = elem.attrib.get("style", "")
        m = _FILL_RE.search(style)
        if m:
            css_fill = m.group(1).strip().lower()
            return css_fill not in ("none", "transparent")
        
        # No fill attribute specified - SVG default is BLACK fill
        return True

    # Iterate over paintable elements
    paintable_tags = {
        "path", "rect", "circle", "ellipse",
        "polygon", "polyline", "line"
    }

    for elem in root.iter():
        tag = elem.tag.split("}")[-1]        # strip namespace
        if tag in paintable_tags and filled(elem):
            # Found at least one shape that is filled → NOT sketch style
            return False

    return True
def is_greyscale(svg_source: str) -> bool:
    """
    Return True if the SVG only uses greyscale colors (black, white, greys).
    
    Args:
        svg_source (str): Raw SVG content as a string
        
    Returns:
        bool: True if the SVG uses only greyscale colors, False otherwise
    """
    try:
        root = ET.fromstring(svg_source)
    except ET.ParseError:
        # Return False for invalid SVG
        return False
    
    # Regular expressions for different color formats
    hex_color_re = re.compile(r'^#([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})$')
    hex_short_color_re = re.compile(r'^#([0-9a-fA-F])([0-9a-fA-F])([0-9a-fA-F])$')
    rgb_color_re = re.compile(r'rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)')
    rgba_color_re = re.compile(r'rgba\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*[\d.]+\s*\)')
    
    # Named greyscale colors
    greyscale_named_colors = {
        'black', 'white', 'gray', 'grey', 'darkgray', 'darkgrey', 
        'dimgray', 'dimgrey', 'lightgray', 'lightgrey', 'gainsboro',
        'silver', 'transparent', 'none'
    }
    
    def is_greyscale(color: str) -> bool:
        """Check if a color value is greyscale."""
        if not color:
            return True
            
        color = color.lower().strip()
        
        # Check for named greyscale colors
        if color in greyscale_named_colors:
            return True
        
        # Check for hex color format (#RRGGBB)
        match = hex_color_re.match(color)
        if match:
            r, g, b = match.groups()
            return r == g == b
        
        # Check for short hex format (#RGB)
        match = hex_short_color_re.match(color)
        if match:
            r, g, b = match.groups()
            return r == g == b
        
        # Check for rgb format
        match = rgb_color_re.match(color)
        if match:
            r, g, b = [int(x) for x in match.groups()]
            return r == g == b
        
        # Check for rgba format
        match = rgba_color_re.match(color)
        if match:
            r, g, b = [int(x) for x in match.groups()]
            return r == g == b
        
        # Unknown color format, conservatively return False
        return False
    
    # Check all elements with potential color attributes
    for elem in root.iter():
        # Check fill attribute
        if 'fill' in elem.attrib and not is_greyscale(elem.attrib['fill']):
            return False
            
        # Check stroke attribute
        if 'stroke' in elem.attrib and not is_greyscale(elem.attrib['stroke']):
            return False
        
        # Check style attribute for colors
        style = elem.attrib.get('style', '')
        
        # Extract fill from style
        fill_match = _FILL_RE.search(style)
        if fill_match and not is_greyscale(fill_match.group(1)):
            return False
            
        # Extract stroke from style
        stroke_match = re.search(r'stroke\s*:\s*([^;]+)', style)
        if stroke_match and not is_greyscale(stroke_match.group(1)):
            return False
    
    return True
if __name__ == "__main__":
    # Test the SVG processing functions
    
    # Test case 1: Sketch-style SVG (no fills)
    sketch_svg = '''
    <svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
        <path d="M10,10 L90,10 L90,90 L10,90 Z" fill="none" stroke="black"/>
        <circle cx="50" cy="50" r="20" stroke="black" fill="transparent"/>
    </svg>
    '''
    
    # Test case 2: Filled SVG (not sketch-style)
    filled_svg = '''
    <svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
        <path d="M10,10 L90,10 L90,90 L10,90 Z" fill="blue" stroke="black"/>
        <circle cx="50" cy="50" r="20" stroke="black" fill="red"/>
    </svg>
    '''
    
    # Test is_sketch_style function
    print("Testing is_sketch_style:")
    print(f"Sketch SVG is sketch style: {is_sketch_style(sketch_svg)}")  # Should be True
    print(f"Filled SVG is sketch style: {is_sketch_style(filled_svg)}")  # Should be False
    
    # Test extract_svg function
    print("\nTesting extract_svg:")
    mixed_content = "Some text before <svg width='100' height='100'><rect width='50' height='50'/></svg> and after"
    extracted = extract_svg(mixed_content)
    print(f"Extracted SVG: {extracted[:40]}...")
    
    # Test rendering functions
    print("\nTesting SVG rendering:")
    image1 = safe_svg_to_image(sketch_svg)
    image2 = safe_svg_to_image(filled_svg)
    print(f"Rendered sketch SVG: {'Success' if image1 is not None else 'Failed'}")
    print(f"Rendered filled SVG: {'Success' if image2 is not None else 'Failed'}")
    
    if image1 and image2:
        # Save the images for inspection
        image1.save("test_sketch.png")
        image2.save("test_filled.png")
        print("Images saved as test_sketch.png and test_filled.png")
    
    
    # Test case 3: Greyscale SVG
    greyscale_svg = '''
    <svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
        <rect x="10" y="10" width="80" height="80" fill="#333333" stroke="black"/>
        <circle cx="50" cy="50" r="30" fill="#999999" stroke="#666"/>
        <path d="M20,20 L80,80" stroke="white" stroke-width="2"/>
    </svg>
    '''

    # Test case 4: Colored SVG
    colored_svg = '''
    <svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
        <rect x="10" y="10" width="80" height="80" fill="red" stroke="black"/>
        <circle cx="50" cy="50" r="30" fill="blue" stroke="#666"/>
        <path d="M20,20 L80,80" stroke="white" stroke-width="2"/>
    </svg>
    '''

    # Test is_greyscale_svg function
    print("\nTesting is_greyscale_svg:")
    print(f"Greyscale SVG is greyscale: {is_greyscale_svg(greyscale_svg)}")  # Should be True
    print(f"Colored SVG is greyscale: {is_greyscale_svg(colored_svg)}")     # Should be False
    print(f"Sketch SVG is greyscale: {is_greyscale_svg(sketch_svg)}")       # Should be True