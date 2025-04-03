import cairosvg
from io import BytesIO
from func_timeout import func_timeout, FunctionTimedOut, func_set_timeout
from lxml import etree
import os
from PIL import Image
import re


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
    ans = text
    match = re.search(r'(<svg .*?</svg>)', ans, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        match = re.search(r'(<svg.*?)', ans, re.DOTALL)
        if match:
            return match.group(1).strip() 
        
        return ""