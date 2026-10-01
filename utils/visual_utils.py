import re
import io
import base64
from PIL import Image, ImageDraw

AGENT_COLORS = {
    1: "red",
    2: "blue",
    3: "green",
    4: "yellow",
    5: "purple",
    6: "orange",
    7: "cyan",
    8: "magenta"
}

def extract_coordinates(text):
    """
    Extracts normalized coordinates from an agent's text response.
    Looks for patterns like [x, y, w, h] or [x, y].
    Returns a list of lists: [[0.1, 0.2, 0.1, 0.1], [0.5, 0.5]]
    """
    coords = []
    # Matches [0.12, 0.34, 0.56, 0.78] or [0.12, 0.34] with optional spaces
    # regex: \[ \s* float \s* , \s* float (\s* , \s* float \s* , \s* float)? \s* \]
    # A float can be matched broadly with: \d*\.?\d+
    pattern = r'\[\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+)(?:\s*,\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+))?\s*\]'
    
    matches = re.finditer(pattern, text)
    for m in matches:
        if m.group(3) is not None and m.group(4) is not None:
            # 4 coordinates
            x, y, w, h = float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))
            coords.append([x, y, w, h])
        else:
            # 2 coordinates
            x, y = float(m.group(1)), float(m.group(2))
            coords.append([x, y])

    # 2. Qwen2.5-VL and Qwen-VL (v1) pattern
    # Matches <|box_start|>(x1,y1),(x2,y2)<|box_end|> or <box>(x1,y1),(x2,y2)</box>
    qwen_pattern = r'(?:<\|box_start\|>|<box>)\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*(?:<\|box_end\|>|</box>)'
    qwen_matches = re.finditer(qwen_pattern, text)
    for m in qwen_matches:
        # Qwen uses a 1000x1000 coordinate grid, normalize to 0.0-1.0
        x1 = float(m.group(1)) / 1000.0
        y1 = float(m.group(2)) / 1000.0
        x2 = float(m.group(3)) / 1000.0
        y2 = float(m.group(4)) / 1000.0
        coords.append([x1, y1, x2, y2])
            
    return coords

def clamp(val):
    return max(0.0, min(1.0, float(val)))

def annotate_image_base64(base64_str, agent_coords_map):
    """
    Takes a base64 encoded image and a dictionary of agent_id -> list of coordinates.
    Draws the coordinates onto the image using the agent's assigned color.
    Returns the new base64 encoded image.
    """
    img_data = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_data)).convert("RGB")
    draw = ImageDraw.Draw(img)
    width, height = img.size
    
    # Responsive sizing
    outline_w = max(2, int(max(width, height) * 0.005))
    point_r = max(4, int(max(width, height) * 0.01))

    for agent_id, coords_list in agent_coords_map.items():
        color = AGENT_COLORS.get(agent_id, "white")
        for coords in coords_list:
            if len(coords) == 4:
                # [x, y, w, h] or [x1, y1, x2, y2]
                val1, val2, val3, val4 = clamp(coords[0]), clamp(coords[1]), clamp(coords[2]), clamp(coords[3])
                abs_x1 = val1 * width
                abs_y1 = val2 * height
                
                # To handle both formats, if val3 + val1 > 1.5, it might be an x2 coordinate rather than width.
                # However, drawing a bounding box between (val1, val2) and (val1+val3, val2+val4) is standard for w, h.
                # If they provided x2, y2, then val3 is x2. We will just draw a rectangle bounded by those two points.
                # We can handle it safely by just finding the min/max X and Y to draw a box.
                if val3 + val1 > 1.05 and val3 <= 1.0:
                    # Likely [x1, y1, x2, y2]
                    abs_x2 = val3 * width
                    abs_y2 = val4 * height
                else:
                    # Likely [x, y, w, h]
                    abs_x2 = (val1 + val3) * width
                    abs_y2 = (val2 + val4) * height
                
                x_min, x_max = min(abs_x1, abs_x2), max(abs_x1, abs_x2)
                y_min, y_max = min(abs_y1, abs_y2), max(abs_y1, abs_y2)
                
                draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=outline_w)
                
            elif len(coords) == 2:
                # [x, y]
                abs_x = clamp(coords[0]) * width
                abs_y = clamp(coords[1]) * height
                draw.ellipse([abs_x - point_r, abs_y - point_r, abs_x + point_r, abs_y + point_r], fill=color)

    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def crop_image_base64_normalized(base64_str, bbox):
    """
    Crops the base64 image using normalized bbox [x_min, y_min, x_max, y_max] (0.0 to 1.0).
    Returns a new base64 string.
    """
    if "base64," in base64_str:
        base64_str = base64_str.split("base64,")[1]
    
    img_data = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_data)).convert("RGB")
    width, height = img.size
    
    x_min, y_min, x_max, y_max = [clamp(v) for v in bbox]
    
    # Optional: add a small padding/margin
    margin = 0.05
    x_min = max(0.0, x_min - margin)
    y_min = max(0.0, y_min - margin)
    x_max = min(1.0, x_max + margin)
    y_max = min(1.0, y_max + margin)

    left = int(x_min * width)
    top = int(y_min * height)
    right = int(x_max * width)
    bottom = int(y_max * height)
    
    # Ensure valid dimensions
    if right <= left or bottom <= top:
        return base64_str # Fallback to original if invalid

    cropped_img = img.crop((left, top, right, bottom))
    
    buffered = io.BytesIO()
    cropped_img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')
