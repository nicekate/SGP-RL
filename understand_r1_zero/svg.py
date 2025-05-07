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



def test_svg_with_definitions():
    """Test svg_to_image function with SVG that uses definitions."""
    
    # Create SVG with various definitions (gradient, pattern, reusable path)
    test_svg = '''
    <svg width="300" height="300" xmlns="http://www.w3.org/2000/svg">
        <!-- Definitions for reuse -->
        <defs>
            <!-- Linear Gradient -->
            <linearGradient id="gradient1" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stop-color="#000000" />
                <stop offset="100%" stop-color="#FFFFFF" />
            </linearGradient>
            
            <!-- Radial Gradient -->
            <radialGradient id="gradient2" cx="50%" cy="50%" r="50%" fx="50%" fy="50%">
                <stop offset="0%" stop-color="#FFFFFF" />
                <stop offset="100%" stop-color="#444444" />
            </radialGradient>
            
            <!-- Pattern -->
            <pattern id="pattern1" x="0" y="0" width="20" height="20" patternUnits="userSpaceOnUse">
                <circle cx="10" cy="10" r="5" fill="black" />
            </pattern>
            
            <!-- Reusable path -->
            <path id="star" d="M10,0 L12.245,6.91 L19.51,6.91 L14.135,11.18 L16.38,18.09 L10,13.82 L3.62,18.09 L5.865,11.18 L0.49,6.91 L7.755,6.91 Z" />
        </defs>
        
        <!-- Background -->
        <rect x="0" y="0" width="300" height="300" fill="#eee" />
        
        <!-- Elements using definitions -->
        <circle cx="75" cy="75" r="50" fill="url(#gradient1)" />
        <circle cx="225" cy="75" r="50" fill="url(#gradient2)" />
        <rect x="25" y="150" width="100" height="100" fill="url(#pattern1)" />
        
        <!-- Reused star shape with different transforms and fills -->
        <g transform="translate(160, 160) scale(3)">
            <use href="#star" fill="black" />
        </g>
        <g transform="translate(210, 210) scale(2)">
            <use href="#star" fill="gray" />
        </g>
        <g transform="translate(240, 180) scale(1.5)">
            <use href="#star" fill="white" stroke="black" />
        </g>
    </svg>
    '''
    
    print("Testing SVG with definitions...")
    
    # Try rendering the SVG
    image = svg_to_image(test_svg)
    
    if image is not None:
        # Save the output image
        output_filename = "test_svg_with_defs.png"
        image.save(output_filename)
        
        # Show image info
        print(f"Successfully rendered SVG to image")
        print(f"Image size: {image.size}")
        print(f"Image mode: {image.mode}")
        print(f"Output saved to: {output_filename}")
        
        return True
    else:
        print("Failed to render SVG with definitions")
        return False

if __name__ == "__main__":
    # Test rendering SVG with definitions
    success = test_svg_with_definitions()
    
    # Test with safe_svg_to_image (with timeout handling)
    print("\nTesting with safe_svg_to_image...")
    
    # Create a simpler SVG for timeout testing
    simple_svg = '''
    <svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stop-color="black"/>
                <stop offset="100%" stop-color="white"/>
            </linearGradient>
        </defs>
        <rect x="10" y="10" width="80" height="80" fill="url(#grad)"/>
    </svg>
    '''
    
    # Test with the safe function
    result = safe_svg_to_image(simple_svg)
    if result is not None:
        result.save("test_safe_svg.png")
        print("safe_svg_to_image test passed - image saved as test_safe_svg.png")
    else:
        print("safe_svg_to_image test failed")