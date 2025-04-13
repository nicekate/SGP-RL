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