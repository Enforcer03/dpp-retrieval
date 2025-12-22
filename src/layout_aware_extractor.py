"""
Layout-Aware Patch Extractor (Fixed & Optimized)
================================================
Fixes:
1. Added 'from PIL import ImageDraw' (Solves NameError)
2. Added 'padding' resizing (Solves Blurry Text)
"""

import numpy as np
from PIL import Image, ImageDraw, ImageOps  # <--- FIXED: Added ImageDraw
from typing import List, Tuple, Dict
from pdf2image import convert_from_path
import torch
import math

class LayoutAwarePatchExtractor:
    def __init__(
        self,
        patch_size: int = 448,  # Increased for readability
        target_patches_per_page: int = 32,
        min_patch_height: int = 60
    ):
        self.patch_size = patch_size
        self.target_patches_per_page = target_patches_per_page
        self.min_patch_height = min_patch_height
    
    def _resize_with_padding(self, img: Image.Image, target_size: int) -> Image.Image:
        """Resizes with white padding (letterboxing) to prevent text distortion."""
        # 1. Create white square background
        new_img = Image.new("RGB", (target_size, target_size), (255, 255, 255))
        
        # 2. Resize source maintaining aspect ratio
        src_w, src_h = img.size
        ratio = min(target_size / src_w, target_size / src_h)
        new_w = int(src_w * ratio)
        new_h = int(src_h * ratio)
        img_resized = img.resize((new_w, new_h), Image.LANCZOS)
        
        # 3. Paste centered
        x_offset = (target_size - new_w) // 2
        y_offset = (target_size - new_h) // 2
        new_img.paste(img_resized, (x_offset, y_offset))
        return new_img

    def extract_from_page(self, page_image: Image.Image):
        """Extract patches using a strict grid."""
        w, h = page_image.size
        target = self.target_patches_per_page
        
        # Force 4 cols x 8 rows for 32 patches (Standard A4 ratio)
        cols = 4 if target == 32 else int(math.sqrt(target * (w/h)))
        rows = target // cols
        
        patches = []
        bboxes = []
        step_x = w / cols
        step_y = h / rows
        
        page_array = np.array(page_image)
        
        for r in range(rows):
            for c in range(cols):
                x1, y1 = int(c * step_x), int(r * step_y)
                x2, y2 = int((c+1) * step_x), int((r+1) * step_y)
                
                # Extract and Pad
                patch = page_array[y1:y2, x1:x2]
                if patch.size == 0: continue
                
                patch_img = Image.fromarray(patch.astype(np.uint8))
                patch_padded = self._resize_with_padding(patch_img, self.patch_size)
                
                patches.append(np.array(patch_padded))
                bboxes.append((x1, y1, x2, y2))
                
        return patches, bboxes

# Wrapper function
def extract_layout_aware_patches(pdf_path, dpi=300, patch_size=448, target_patches_per_page=32):
    print(f"\n📄 Extracting patches from: {pdf_path}")
    try:
        pages = convert_from_path(pdf_path, dpi=dpi)
    except Exception as e:
        print(f"❌ Error: {e}")
        return torch.empty(0), []
        
    extractor = LayoutAwarePatchExtractor(patch_size, target_patches_per_page)
    all_patches, all_metadata = [], []
    
    for i, page in enumerate(pages):
        print(f"   Processing page {i+1}/{len(pages)}...", end='\r')
        patches, bboxes = extractor.extract_from_page(page)
        for p, b in zip(patches, bboxes):
            all_patches.append(p)
            all_metadata.append({'page_idx': i, 'bbox': b, 'page_size': page.size})
            
    print(f"\n   ✓ Extracted {len(all_patches)} patches.")
    if not all_patches: return torch.empty(0), []
    
    return torch.from_numpy(np.stack(all_patches)).float().permute(0,3,1,2) / 255.0, all_metadata

class LayoutAwareHighlighter:
    def __init__(self, pdf_path, dpi=300):
        self.pages = convert_from_path(pdf_path, dpi=dpi)
        self.dpi = dpi

    def highlight_patches(self, selected_indices, metadata, output_path, box_color=(255,0,0), box_width=4, opacity=0.2, show_labels=True, scores=None):
        print(f"\n🎨 Highlighting {len(selected_indices)} patches...")
        patches_by_page = {}
        for idx in selected_indices:
            meta = metadata[idx]
            patches_by_page.setdefault(meta['page_idx'], []).append(meta['bbox'])

        result_pages = []
        for i, page in enumerate(self.pages):
            if i in patches_by_page:
                # FIXED: ImageDraw is now imported
                overlay = Image.new('RGBA', page.size, (255,255,255,0))
                draw = ImageDraw.Draw(overlay) 
                for (x1,y1,x2,y2) in patches_by_page[i]:
                    draw.rectangle([x1,y1,x2,y2], fill=box_color+(50,), outline=box_color+(255,), width=4)
                page = Image.alpha_composite(page.convert('RGBA'), overlay).convert('RGB')
            result_pages.append(page)
            
        result_pages[0].save(output_path, save_all=True, append_images=result_pages[1:], resolution=self.dpi)
        print(f"   ✓ Saved: {output_path}")




# """
# Layout-Aware Patch Extractor
# =============================

# Extracts coherent patches that respect document structure:
# - Text columns (left/right)
# - Paragraph boundaries
# - Section headers
# - Figures and tables
# """

# import numpy as np
# from PIL import Image, ImageDraw
# from typing import List, Tuple, Dict
# from pdf2image import convert_from_path
# import torch


# class LayoutAwarePatchExtractor:
#     """
#     Extracts patches intelligently based on document layout.
    
#     Instead of a dumb grid, this:
#     1. Detects columns using vertical projection
#     2. Finds horizontal gaps (paragraph breaks)
#     3. Creates patches that align with content boundaries
#     """
    
#     def __init__(
#         self,
#         patch_size: int = 224,
#         target_patches_per_page: int = 8,
#         min_patch_height: int = 60  # Reduced from 80
#     ):
#         self.patch_size = patch_size
#         self.target_patches_per_page = target_patches_per_page
#         self.min_patch_height = min_patch_height
    
#     def detect_columns(self, page_array: np.ndarray, threshold_percentile: int = 15) -> List[Tuple[int, int]]:
#         """
#         Detect text columns using vertical projection.
        
#         Strategy:
#         1. Convert to grayscale
#         2. Sum pixels vertically (vertical projection)
#         3. Find valleys (white space between columns)
#         4. Return column boundaries
#         """
#         # Convert to grayscale
#         if len(page_array.shape) == 3:
#             gray = np.mean(page_array, axis=2)
#         else:
#             gray = page_array
        
#         # Invert (dark text on light bg → bright text on dark bg)
#         gray_inv = 255 - gray
        
#         # Vertical projection (sum along rows)
#         v_proj = np.sum(gray_inv, axis=0)
        
#         # Smooth to reduce noise
#         window = 15
#         v_proj_smooth = np.convolve(v_proj, np.ones(window)/window, mode='same')
        
#         # Find threshold for gaps
#         threshold = np.percentile(v_proj_smooth, threshold_percentile)
        
#         # Find column boundaries
#         is_gap = v_proj_smooth < threshold
        
#         columns = []
#         in_column = False
#         col_start = 0
        
#         min_column_width = 100  # Minimum pixels for valid column
        
#         for x in range(len(is_gap)):
#             if not is_gap[x] and not in_column:
#                 # Start of column
#                 col_start = x
#                 in_column = True
#             elif is_gap[x] and in_column:
#                 # End of column
#                 if x - col_start > min_column_width:
#                     columns.append((col_start, x))
#                 in_column = False
        
#         # Handle last column
#         if in_column and len(v_proj_smooth) - col_start > min_column_width:
#             columns.append((col_start, len(v_proj_smooth)))
        
#         # If no columns detected, treat whole page as one column
#         if len(columns) == 0:
#             # Use full width with small margins
#             margin = 50
#             columns = [(margin, gray.shape[1] - margin)]
        
#         return columns
    
#     def find_horizontal_gaps(
#         self,
#         page_array: np.ndarray,
#         col_left: int,
#         col_right: int,
#         gap_threshold_percentile: int = 35  # Increased from 25 to find more breaks
#     ) -> List[int]:
#         """
#         Find horizontal gaps (paragraph breaks) within a column.
#         """
#         # Extract column
#         if len(page_array.shape) == 3:
#             gray = np.mean(page_array[:, col_left:col_right], axis=2)
#         else:
#             gray = page_array[:, col_left:col_right]
        
#         # Invert
#         gray_inv = 255 - gray
        
#         # Horizontal projection
#         h_proj = np.sum(gray_inv, axis=1)
        
#         # Smooth
#         window = 3  # Reduced from 5 for more sensitivity
#         h_proj_smooth = np.convolve(h_proj, np.ones(window)/window, mode='same')
        
#         # Find gaps
#         threshold = np.percentile(h_proj_smooth, gap_threshold_percentile)
#         is_gap = h_proj_smooth < threshold
        
#         # Find significant gaps (reduced minimum size)
#         gaps = []
#         gap_start = -1
#         min_gap_size = 5  # Reduced from 10 for more breaks
        
#         for y in range(len(is_gap)):
#             if is_gap[y] and gap_start == -1:
#                 gap_start = y
#             elif not is_gap[y] and gap_start != -1:
#                 gap_size = y - gap_start
#                 if gap_size >= min_gap_size:
#                     # Use middle of gap
#                     gaps.append(gap_start + gap_size // 2)
#                 gap_start = -1
        
#         return gaps
    
#     def create_patches(
#         self,
#         page_array: np.ndarray,
#         columns: List[Tuple[int, int]]
#     ) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
#         """
#         Create patches from columns and horizontal boundaries.
        
#         Enforces maximum patch height to prevent entire pages being selected.
        
#         Returns:
#             patches: List of patch arrays
#             bboxes: List of (x1, y1, x2, y2) bounding boxes
#         """
#         patches = []
#         bboxes = []
        
#         h, w = page_array.shape[:2]
        
#         # Maximum patch height (in pixels) - prevents entire page selection
#         # Scaled for DPI: 600px at 300 DPI ≈ 2 inches = reasonable patch
#         max_patch_height = 600  # Doubled from 300 for 300 DPI
#         overlap = 40  # Increased overlap for higher resolution
        
#         for col_left, col_right in columns:
#             # Find horizontal gaps in this column
#             gaps = self.find_horizontal_gaps(page_array, col_left, col_right)
            
#             # Add page boundaries
#             boundaries = [0] + sorted(gaps) + [h]
            
#             # Create patches between boundaries
#             for i in range(len(boundaries) - 1):
#                 y1 = boundaries[i]
#                 y2 = boundaries[i + 1]
                
#                 patch_height = y2 - y1
                
#                 # Skip if too small
#                 if patch_height < self.min_patch_height:
#                     continue
                
#                 # CRITICAL: Force split if too large
#                 if patch_height > max_patch_height:
#                     # Split into multiple patches with small overlap
#                     num_splits = int(np.ceil(patch_height / max_patch_height))
#                     split_height = patch_height // num_splits
                    
#                     for split_idx in range(num_splits):
#                         split_y1 = y1 + split_idx * split_height
#                         split_y2 = min(y1 + (split_idx + 1) * split_height + overlap, y2)
                        
#                         if split_y2 - split_y1 < self.min_patch_height:
#                             continue
                        
#                         # Extract patch
#                         patch = page_array[split_y1:split_y2, col_left:col_right]
                        
#                         # Resize to standard size
#                         patch_img = Image.fromarray(patch.astype(np.uint8))
#                         patch_resized = patch_img.resize((self.patch_size, self.patch_size), Image.LANCZOS)
                        
#                         patches.append(np.array(patch_resized))
#                         bboxes.append((col_left, split_y1, col_right, split_y2))
#                 else:
#                     # Extract patch directly
#                     patch = page_array[y1:y2, col_left:col_right]
                    
#                     # Resize to standard size
#                     patch_img = Image.fromarray(patch.astype(np.uint8))
#                     patch_resized = patch_img.resize((self.patch_size, self.patch_size), Image.LANCZOS)
                    
#                     patches.append(np.array(patch_resized))
#                     bboxes.append((col_left, y1, col_right, y2))
        
#         return patches, bboxes
    
#     def extract_from_page(
#         self,
#         page_image: Image.Image
#     ) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
#         """
#         Extract layout-aware patches from a single page.
        
#         Guarantees minimum number of patches per page.
#         """
#         # Convert to array
#         page_array = np.array(page_image)
        
#         # Detect columns
#         columns = self.detect_columns(page_array)
        
#         # Create patches
#         patches, bboxes = self.create_patches(page_array, columns)
        
#         # CRITICAL: Ensure minimum patches per page
#         # If too few patches detected, add uniform splits
#         if len(patches) < self.target_patches_per_page // 2:
#             print(f"   ⚠️  Only {len(patches)} patches detected, adding uniform splits...")
#             patches, bboxes = self._add_uniform_splits(page_array, columns, patches, bboxes)
        
#         # Subsample if too many patches
#         if len(patches) > self.target_patches_per_page * 1.5:
#             # Keep evenly spaced patches
#             indices = np.linspace(0, len(patches) - 1, self.target_patches_per_page, dtype=int)
#             patches = [patches[i] for i in indices]
#             bboxes = [bboxes[i] for i in indices]
        
#         return patches, bboxes
    
#     def _add_uniform_splits(
#         self,
#         page_array: np.ndarray,
#         columns: List[Tuple[int, int]],
#         existing_patches: List[np.ndarray],
#         existing_bboxes: List[Tuple[int, int, int, int]]
#     ) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
#         """
#         Add uniform vertical splits to ensure minimum patches.
#         """
#         h, w = page_array.shape[:2]
#         target_count = self.target_patches_per_page
        
#         # Calculate how many more patches needed
#         needed = max(0, target_count - len(existing_patches))
        
#         if needed == 0:
#             return existing_patches, existing_bboxes
        
#         # Create uniform splits across columns
#         patches = list(existing_patches)
#         bboxes = list(existing_bboxes)
        
#         splits_per_column = needed // len(columns) + 1
        
#         for col_left, col_right in columns:
#             # Create uniform vertical splits
#             split_height = h // splits_per_column
            
#             for i in range(splits_per_column):
#                 y1 = i * split_height
#                 y2 = min((i + 1) * split_height, h)
                
#                 if y2 - y1 < self.min_patch_height:
#                     continue
                
#                 # Check if this overlaps significantly with existing patches
#                 bbox = (col_left, y1, col_right, y2)
#                 if self._overlaps_existing(bbox, existing_bboxes):
#                     continue
                
#                 # Extract patch
#                 patch = page_array[y1:y2, col_left:col_right]
#                 patch_img = Image.fromarray(patch.astype(np.uint8))
#                 patch_resized = patch_img.resize((self.patch_size, self.patch_size), Image.LANCZOS)
                
#                 patches.append(np.array(patch_resized))
#                 bboxes.append(bbox)
                
#                 if len(patches) >= target_count:
#                     break
            
#             if len(patches) >= target_count:
#                 break
        
#         return patches, bboxes
    
#     def _overlaps_existing(
#         self,
#         new_bbox: Tuple[int, int, int, int],
#         existing_bboxes: List[Tuple[int, int, int, int]],
#         threshold: float = 0.5
#     ) -> bool:
#         """Check if new bbox significantly overlaps with existing ones."""
#         x1_new, y1_new, x2_new, y2_new = new_bbox
#         area_new = (x2_new - x1_new) * (y2_new - y1_new)
        
#         for x1_ex, y1_ex, x2_ex, y2_ex in existing_bboxes:
#             # Calculate intersection
#             x1_int = max(x1_new, x1_ex)
#             y1_int = max(y1_new, y1_ex)
#             x2_int = min(x2_new, x2_ex)
#             y2_int = min(y2_new, y2_ex)
            
#             if x1_int < x2_int and y1_int < y2_int:
#                 area_int = (x2_int - x1_int) * (y2_int - y1_int)
#                 overlap_ratio = area_int / area_new
                
#                 if overlap_ratio > threshold:
#                     return True
        
#         return False


# def extract_layout_aware_patches(
#     pdf_path: str,
#     dpi: int = 300,  # Increased from 150 for better quality
#     patch_size: int = 448,  # Doubled from 224 for more detail
#     target_patches_per_page: int = 10
# ) -> Tuple[torch.Tensor, List[Dict]]:
#     """
#     Extract layout-aware patches from PDF.
    
#     Args:
#         pdf_path: Path to PDF
#         dpi: Resolution for rendering (300 = high quality)
#         patch_size: Size of output patches (448 = 2x standard)
#         target_patches_per_page: Target number of patches per page
    
#     Returns:
#         patches_tensor: Tensor of shape (N, 3, H, W) normalized to [0, 1]
#         metadata: List of dicts with bbox and page info for each patch
#     """
#     print(f"\n📄 Extracting layout-aware patches from: {pdf_path}")
#     print(f"   Resolution: {dpi} DPI, Patch size: {patch_size}×{patch_size}")
    
#     # Convert PDF to images
#     pages = convert_from_path(pdf_path, dpi=dpi)
#     print(f"   ✓ Loaded {len(pages)} pages")
    
#     # Initialize extractor with stricter parameters
#     extractor = LayoutAwarePatchExtractor(
#         patch_size=patch_size,
#         target_patches_per_page=target_patches_per_page,
#         min_patch_height=120  # Scaled with DPI (60 * 2)
#     )
    
#     all_patches = []
#     all_metadata = []
#     patch_heights = []  # Track patch heights for diagnostics
    
#     for page_idx, page_image in enumerate(pages):
#         print(f"   Processing page {page_idx + 1}/{len(pages)}...", end='\r')
        
#         # Extract patches
#         patches, bboxes = extractor.extract_from_page(page_image)
        
#         # Store with metadata and track sizes
#         for patch, bbox in zip(patches, bboxes):
#             all_patches.append(patch)
            
#             x1, y1, x2, y2 = bbox
#             patch_height_pixels = y2 - y1
#             patch_heights.append(patch_height_pixels)
            
#             all_metadata.append({
#                 'page_idx': page_idx,
#                 'bbox': bbox,
#                 'page_size': page_image.size,
#                 'patch_height': patch_height_pixels
#             })
    
#     print(f"\n   ✓ Extracted {len(all_patches)} layout-aware patches")
#     print(f"   • Avg per page: {len(all_patches)/len(pages):.1f}")
#     print(f"   • Patch height range: {min(patch_heights):.0f} - {max(patch_heights):.0f} pixels")
#     print(f"   • Avg patch height: {np.mean(patch_heights):.0f} pixels")
    
#     # Warn if patches are too large (scaled with DPI)
#     large_patches = [h for h in patch_heights if h > 800]  # 400 * 2
#     if large_patches:
#         print(f"   ⚠️  {len(large_patches)} patches are >800px tall (may be too large)")
    
#     # Convert to tensor
#     patches_array = np.stack(all_patches)
#     patches_tensor = torch.from_numpy(patches_array).float() / 255.0
#     patches_tensor = patches_tensor.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
    
#     return patches_tensor, all_metadata


# # Update PDFHighlighter to use exact bboxes from metadata
# class LayoutAwareHighlighter:
#     """
#     Highlights patches using exact bounding boxes from extraction.
#     """
    
#     def __init__(self, pdf_path: str, dpi: int = 300):  # Increased from 150
#         self.pdf_path = pdf_path
#         self.dpi = dpi
#         self.pages = convert_from_path(pdf_path, dpi=dpi)
    
#     def highlight_patches(
#         self,
#         selected_indices: List[int],
#         metadata: List[Dict],
#         output_path: str,
#         box_color: Tuple[int, int, int] = (255, 0, 0),
#         box_width: int = 4,
#         opacity: float = 0.2,
#         show_labels: bool = True,
#         scores: List[float] = None
#     ) -> str:
#         """
#         Highlight selected patches using exact bounding boxes.
#         """
#         print(f"\n🎨 Highlighting {len(selected_indices)} patches...")
        
#         # Group by page
#         patches_by_page = {}
#         for idx, patch_idx in enumerate(selected_indices):
#             meta = metadata[patch_idx]
#             page_idx = meta['page_idx']
            
#             if page_idx not in patches_by_page:
#                 patches_by_page[page_idx] = []
            
#             patches_by_page[page_idx].append({
#                 'bbox': meta['bbox'],
#                 'order': idx + 1,
#                 'score': scores[idx] if scores else None
#             })
        
#         # Create highlighted pages
#         highlighted_pages = []
        
#         for page_idx, page in enumerate(self.pages):
#             page_rgba = page.convert('RGBA')
#             overlay = Image.new('RGBA', page_rgba.size, (255, 255, 255, 0))
#             draw = ImageDraw.Draw(overlay)
            
#             if page_idx in patches_by_page:
#                 for patch_info in patches_by_page[page_idx]:
#                     x1, y1, x2, y2 = patch_info['bbox']
                    
#                     # Semi-transparent fill
#                     fill_color = box_color + (int(255 * opacity),)
#                     draw.rectangle([x1, y1, x2, y2], fill=fill_color)
                    
#                     # Solid border
#                     border_color = box_color + (255,)
#                     for i in range(box_width):
#                         draw.rectangle([x1+i, y1+i, x2-i, y2-i], outline=border_color)
                    
#                     # Label
#                     if show_labels:
#                         from PIL import ImageFont
#                         try:
#                             font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
#                         except:
#                             font = ImageFont.load_default()
                        
#                         label = f"#{patch_info['order']}"
#                         if patch_info['score']:
#                             label += f"\n{patch_info['score']:.2f}"
                        
#                         # White background for label
#                         bbox = draw.textbbox((x1 + 5, y1 + 5), label, font=font)
#                         draw.rectangle(bbox, fill=(255, 255, 255, 230))
#                         draw.text((x1 + 5, y1 + 5), label, fill=border_color, font=font)
            
#             # Composite
#             highlighted = Image.alpha_composite(page_rgba, overlay).convert('RGB')
#             highlighted_pages.append(highlighted)
        
#         # Save as PDF
#         if highlighted_pages:
#             highlighted_pages[0].save(
#                 output_path,
#                 save_all=True,
#                 append_images=highlighted_pages[1:],
#                 resolution=self.dpi,
#                 quality=95
#             )
#             print(f"   ✓ Saved: {output_path}")
        
#         return output_path


# if __name__ == "__main__":
#     print("Layout-Aware Patch Extractor")
#     print("Extracts patches that respect document structure.")