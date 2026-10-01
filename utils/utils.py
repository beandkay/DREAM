import os
import json
import base64
import warnings
from copy import deepcopy

def is_multimodal_dataset(sample):
    """Check if a sample contains multimodal data (has image_path field)"""
    return "image_path" in sample

def encode_image_to_base64(image_path):
    """Encode image file to base64 string"""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string
    except FileNotFoundError:
        warnings.warn(f"Image file not found: {image_path}", category=UserWarning, stacklevel=3)
        return None
    except PermissionError:
        warnings.warn(f"Permission denied accessing image file: {image_path}", category=UserWarning, stacklevel=3)
        return None
    except Exception as e:
        warnings.warn(f"Error encoding image {image_path}: {e}", category=UserWarning, stacklevel=3)
        return None

def read_valid_jsonl(file_name):
    all_data = []
    with open(file_name, "r") as f:
        tmp = f.readlines()
    for line in tmp:
        try:
            all_data.append(json.loads(line))
        except Exception as e:
            print(line)
            print(f"{e}")
    return all_data

def prepare_multimodal_content(sample):
    """Prepare multimodal content for LLM input"""
    if not is_multimodal_dataset(sample):
        return sample["query"]
    
    # Encode image to base64
    image_base64 = encode_image_to_base64(sample["image_path"])
    if image_base64 is None:
        raise FileNotFoundError(f"Failed to encode image: {sample['image_path']}")
    
    # Create multimodal content
    multimodal_content = [
        {
            "type": "text",
            "text": sample["query"]
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}"
            }
        }
    ]
    
    return multimodal_content

def prepare_qwen_augmented_content(sample):
    """Prepare 5 augmented image contents for Qwen API with actual pixel-level transforms.

    Produces: 1x original (raw file bytes), 2x upscale, 0.5x downscale, h-flip, v-flip.
    """
    if not is_multimodal_dataset(sample):
        return sample["query"]
        
    image_path = sample["image_path"]
    try:
        from PIL import Image

        # Use raw file bytes for the 1x original to avoid re-encoding artifacts
        base64_orig = encode_image_to_base64(image_path)
        if base64_orig is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            orig_width = img.width
            orig_height = img.height

            # 2x upscale — actual pixel resize
            img_2x = img.resize((orig_width * 2, orig_height * 2), Image.LANCZOS)
            base64_2x = encode_pil_image_to_base64(img_2x)

            # 0.5x downscale — actual pixel resize
            img_half = img.resize((max(1, orig_width // 2), max(1, orig_height // 2)), Image.LANCZOS)
            base64_half = encode_pil_image_to_base64(img_half)

            # Flips
            hflip = img.transpose(Image.FLIP_LEFT_RIGHT)
            vflip = img.transpose(Image.FLIP_TOP_BOTTOM)
            base64_hflip = encode_pil_image_to_base64(hflip)
            base64_vflip = encode_pil_image_to_base64(vflip)
    except Exception as e:
        warnings.warn(f"Failed to generate image augmentations: {e}")
        return prepare_multimodal_content(sample)  # fallback

    query = sample["query"]
    query += "\n\nNote: You are provided with 5 variations of the same image: an original resolution version, a 2x scaled version, a 0.5x scaled version, a horizontally flipped version, and a vertically flipped version. Please use all evidence carefully, and be aware that spatial directions (left/right, up/down) in the flipped images will contradict the original image."

    content = [{"type": "text", "text": query}]
    
    # 1. Original (1x) — raw file bytes, no re-encoding
    content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{base64_orig}"}
    })
    # 2. 2x Scale — actual pixel resize
    content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{base64_2x}"}
    })
    # 3. 0.5x Scale — actual pixel resize
    content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{base64_half}"}
    })
    # 4. H-Flip (1x)
    content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{base64_hflip}"}
    })
    # 5. V-Flip (1x)
    content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{base64_vflip}"}
    })
    
    return content

def compose_multimodal_input(text, multimodal_content):
    """Attach text to multimodal template so downstream calls can include images."""
    if multimodal_content is None:
        return text

    payload = deepcopy(multimodal_content)
    text_slot_found = False
    for part in payload:
        if isinstance(part, dict) and part.get("type") == "text":
            part["text"] = text
            text_slot_found = True
            break

    if not text_slot_found:
        payload.insert(0, {"type": "text", "text": text})

    return payload

def transform_image_payload(multimodal_content, resolutions=None, flips=None,
                           shifts=None, crops=None, blurs=None, grayscale=None):
    """Transforms a multimodal content array by applying image augmentations.

    All transforms are applied at the pixel level so the model actually
    receives different image data.  Supported transforms (applied in order):

        - flips: "hflip" (horizontal) or "vflip" (vertical)
        - shifts: "left10", "right20", "up15", "down10" (translate by N% in direction)
        - crops: 0.8 (center crop to 80% of original size)
        - blurs: 3 (Gaussian blur with radius 3)
        - grayscale: True (convert to grayscale)
        - resolutions: "0.5x", "2x" (scale factor) or "640x480" (absolute size)
                       "1x" or None is treated as identity (no resize).

    If any parameter is a list, generates all specified variations and appends
    them to the payload.
    """
    if multimodal_content is None:
        return None

    import base64
    import io
    import re as _re
    from PIL import Image, ImageFilter
    from copy import deepcopy

    # ---- Helper: parse a resolution string into (target_w, target_h) or None ----
    def _parse_resolution(res_val, src_w, src_h):
        """Return (width, height) to resize to, or None for identity."""
        if res_val is None:
            return None
        res_str = str(res_val).lower().strip()
        if res_str.endswith("x") and res_str[:-1].replace('.', '', 1).isdigit():
            scale = float(res_str[:-1])
            if scale == 1.0:
                return None  # 1x is identity — no resize needed
            return (max(1, int(src_w * scale)), max(1, int(src_h * scale)))
        parts = res_str.split("x")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            tw, th = int(parts[0]), int(parts[1])
            if tw == src_w and th == src_h:
                return None  # already at target size
            return (tw, th)
        return None

    # Check if any transform is specified
    has_transform = any([resolutions, flips, shifts, crops, blurs, grayscale])
    if not has_transform:
        return deepcopy(multimodal_content)

    # Standardize all params to lists
    def _to_list(v):
        if v is None:
            return [None]
        return v if isinstance(v, list) else [v]

    resolutions = _to_list(resolutions)
    flips = _to_list(flips)
    shifts = _to_list(shifts)
    crops = _to_list(crops)
    blurs = _to_list(blurs)
    grayscale = _to_list(grayscale)

    length = max(len(resolutions), len(flips), len(shifts), len(crops), len(blurs), len(grayscale))

    # Pad shorter lists to match length
    def _pad(lst, n):
        return lst * n if len(lst) == 1 else lst + [lst[-1]] * max(0, n - len(lst))

    resolutions = _pad(resolutions, length)
    flips = _pad(flips, length)
    shifts = _pad(shifts, length)
    crops = _pad(crops, length)
    blurs = _pad(blurs, length)
    grayscale = _pad(grayscale, length)

    # Separate text/other items from image items, preserving insertion points
    # so we can restore the original ordering.
    new_content = []           # will hold the final output
    orig_image_indices = []    # positions in new_content where image groups go
    orig_images = []           # original image items (with raw url bytes)
    for item in multimodal_content:
        if item.get("type") == "image_url":
            orig_image_indices.append(len(new_content))
            new_content.append(None)  # placeholder
            orig_images.append(deepcopy(item))
        else:
            new_content.append(deepcopy(item))

    if not orig_images:
        return deepcopy(multimodal_content)

    # Build transformed image items for each variation index
    # image_groups[img_idx] = list of transformed items across all variations
    image_groups = [[] for _ in orig_images]

    for idx in range(length):
        res = resolutions[idx]
        flip = flips[idx]
        shift = shifts[idx]
        crop = crops[idx]
        blur_val = blurs[idx]
        gs = grayscale[idx]

        for img_i, orig_img in enumerate(orig_images):
            item = deepcopy(orig_img)
            url_data = item["image_url"]["url"]

            if "base64," not in url_data:
                # Non-base64 URL — can't transform pixels, pass through
                image_groups[img_i].append(item)
                continue

            prefix, b64_str = url_data.split("base64,", 1)
            img_data = base64.b64decode(b64_str)

            # Determine what pixel work is needed
            needs_augment = flip or shift or crop or blur_val or gs

            # Peek at image dimensions to check if resolution is truly a resize
            if res is not None:
                tmp_img = Image.open(io.BytesIO(img_data))
                src_w, src_h = tmp_img.size
                tmp_img.close()
                target_size = _parse_resolution(res, src_w, src_h)
            else:
                target_size = None

            needs_resize = target_size is not None

            if not needs_augment and not needs_resize:
                # Identity — preserve original bytes exactly
                image_groups[img_i].append(item)
                continue

            # Decode image for pixel transforms
            img = Image.open(io.BytesIO(img_data)).convert("RGB")

            # 1. Flip
            if flip:
                if flip.lower() == "hflip":
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                elif flip.lower() == "vflip":
                    img = img.transpose(Image.FLIP_TOP_BOTTOM)

            # 2. Shift / Translate (e.g. "right10" = shift right by 10%)
            if shift:
                m = _re.match(r'(left|right|up|down)(\d+)', str(shift).lower().strip())
                if m:
                    direction, pct = m.group(1), int(m.group(2))
                    w, h = img.size
                    dx, dy = 0, 0
                    if direction == "left":
                        dx = -int(w * pct / 100)
                    elif direction == "right":
                        dx = int(w * pct / 100)
                    elif direction == "up":
                        dy = -int(h * pct / 100)
                    elif direction == "down":
                        dy = int(h * pct / 100)
                    img = img.transform(img.size, Image.AFFINE, (1, 0, -dx, 0, 1, -dy), fillcolor=(0, 0, 0))

            # 3. Center crop (e.g. 0.8 = crop to 80% centered)
            if crop:
                crop_ratio = float(crop)
                w, h = img.size
                new_w, new_h = int(w * crop_ratio), int(h * crop_ratio)
                left_x = (w - new_w) // 2
                top_y = (h - new_h) // 2
                img = img.crop((left_x, top_y, left_x + new_w, top_y + new_h))

            # 4. Gaussian blur
            if blur_val:
                img = img.filter(ImageFilter.GaussianBlur(radius=int(blur_val)))

            # 5. Grayscale
            if gs:
                img = img.convert("L").convert("RGB")

            # 6. Resolution — actual pixel resize
            if needs_resize:
                img = img.resize(target_size, Image.LANCZOS)

            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=100, subsampling=0)
            new_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            item["image_url"] = {"url": f"{prefix}base64,{new_b64}"}

            image_groups[img_i].append(item)

    # Re-assemble: replace placeholders with transformed image groups,
    # preserving the original text ↔ image ordering.
    assembled = []
    for pos, item in enumerate(new_content):
        if item is None:
            # This was an image placeholder — find which image index
            img_idx = orig_image_indices.index(pos)
            assembled.extend(image_groups[img_idx])
        else:
            assembled.append(item)

    # Update text prompt if we are generating multiple variations
    if length > 1:
        note = f"\n\nNote: You are provided with {length} variations of the image. Please use all evidence carefully."
        for item in assembled:
            if item.get("type") == "text":
                item["text"] = item.get("text", "") + note
                break

    return assembled

def load_model_api_config(config_path, model_name=None):
    with open(config_path, "r") as f:
        api_config = json.load(f)
    for m_name in api_config:
        actual_max_workers = api_config[m_name]["max_workers_per_model"] * len(api_config[m_name]["model_list"])
        api_config[m_name]["max_workers"] = actual_max_workers
    return api_config

def write_to_jsonl(lock, file_name, data):
    with lock:
        with open(file_name, 'a') as f:
            json.dump(data, f)
            f.write('\n')

def reserve_unprocessed_queries(output_path, test_dataset):
    processed_queries = set()
    keep_lines = []
    needs_rewrite = False
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                infered_sample = json.loads(line)
                response = infered_sample.get("response")
                has_error = bool(infered_sample.get("error"))
                if response is None or has_error:
                    needs_rewrite = True
                    continue
                processed_queries.add(infered_sample.get("query"))
                keep_lines.append(line)

    if needs_rewrite and keep_lines:
        with open(output_path, "w") as f:
            f.writelines(keep_lines)

    test_dataset = [sample for sample in test_dataset if sample["query"] not in processed_queries]
    return test_dataset


# ---------------------------------------------------------------------------
# Multi-image & text-rendering utilities (used by RELD-V and VCD-Debate)
# ---------------------------------------------------------------------------

def encode_pil_image_to_base64(pil_image):
    """Encode a PIL Image object directly to a base64 string (JPEG)."""
    import io as _io
    buffer = _io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=100, subsampling=0)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def build_multi_image_content(text, image_base64_list):
    """Build an OpenAI-compatible multimodal content array with multiple images.

    Parameters
    ----------
    text : str
        The text portion of the message.
    image_base64_list : list[str]
        One or more base64-encoded image strings.

    Returns
    -------
    list[dict]
        A content array: [{"type": "text", ...}, {"type": "image_url", ...}, ...]
    """
    content = [{"type": "text", "text": text}]
    for img_b64 in image_base64_list:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
        })
    return content


def render_text_to_image(text, width=800, font_size=20, distort=False,
                         distortion_strength=0.3):
    """Render a text string into an image and return its base64 encoding.

    The image has a white background with black text, wrapped to fit *width*.
    When *distort* is ``True`` a subtle sinusoidal wave transform is applied
    (inspired by the LoMo paper) to prevent trivial OCR bypass.

    Parameters
    ----------
    text : str
        The text to render.
    width : int
        Image width in pixels.
    font_size : int
        Font size in pixels.
    distort : bool
        Whether to apply wave distortion.
    distortion_strength : float
        Controls wave amplitude (0.0 – 1.0).

    Returns
    -------
    str
        Base64-encoded JPEG string of the rendered image.
    """
    from PIL import Image, ImageDraw, ImageFont
    import textwrap, math

    # Determine characters per line based on rough monospace estimate
    chars_per_line = max(1, (width - 40) // (font_size * 6 // 10))
    wrapped = textwrap.fill(text, width=chars_per_line)
    lines = wrapped.split("\n")

    line_height = font_size + 6
    height = max(60, line_height * len(lines) + 40)

    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Try to use a clean font; fall back to default
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

    y = 20
    for line in lines:
        draw.text((20, y), line, fill=(0, 0, 0), font=font)
        y += line_height

    # Optional wave distortion
    if distort:
        import numpy as np
        arr = np.array(img)
        amplitude = distortion_strength * 8
        period = height / 2.0
        for row in range(arr.shape[0]):
            shift = int(amplitude * math.sin(2 * math.pi * row / period))
            arr[row] = np.roll(arr[row], shift, axis=0)
        img = Image.fromarray(arr)

    return encode_pil_image_to_base64(img)


def split_image_quadrants(image_path):
    """Split an image into 4 equal quadrants (2×2 grid).

    Parameters
    ----------
    image_path : str
        Path to the image file.

    Returns
    -------
    dict[str, str]
        Mapping of position label to base64-encoded JPEG string.
        Keys: "top-left", "top-right", "bottom-left", "bottom-right".
    """
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    mid_w, mid_h = w // 2, h // 2

    quadrants = {
        "top-left":     img.crop((0, 0, mid_w, mid_h)),
        "top-right":    img.crop((mid_w, 0, w, mid_h)),
        "bottom-left":  img.crop((0, mid_h, mid_w, h)),
        "bottom-right": img.crop((mid_w, mid_h, w, h)),
    }

    return {pos: encode_pil_image_to_base64(quad) for pos, quad in quadrants.items()}