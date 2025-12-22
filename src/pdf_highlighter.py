"""
PDF Highlighter for Budgeted Dual-DPP
======================================

Highlights selected patches on the original PDF with bounding boxes.
"""

import os
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pdf2image import convert_from_path


class PDFHighlighter:
    """
    Highlights selected patches on PDF pages with colored bounding boxes.
    """
    
    def __init__(
        self,
        pdf_path: str,
        patches_per_page: int = 12,
        dpi: int = 150
    ):
        """
        Initialize PDF highlighter.
        
        Args:
            pdf_path: Path to PDF file
            patches_per_page: Number of patches per page (must match extraction)
            dpi: DPI for PDF rendering
        """
        self.pdf_path = pdf_path
        self.patches_per_page = patches_per_page
        self.dpi = dpi
        
        # Convert PDF to images
        print(f"\n🖼️  Loading PDF for highlighting: {pdf_path}")
        self.pages = convert_from_path(pdf_path, dpi=dpi)
        print(f"   ✓ Loaded {len(self.pages)} pages")
        
        # Calculate grid dimensions
        self.grid_h = int(np.sqrt(patches_per_page))
        self.grid_w = patches_per_page // self.grid_h
        
        # Ensure grid matches patches
        while self.grid_h * self.grid_w < patches_per_page:
            self.grid_w += 1
    
    def _get_patch_bbox(self, patch_idx: int) -> Tuple[int, int, int, int, int]:
        """
        Get bounding box coordinates for a patch.
        
        Args:
            patch_idx: Global patch index
        
        Returns:
            Tuple of (page_idx, x1, y1, x2, y2)
        """
        # Calculate which page this patch is on
        page_idx = patch_idx // self.patches_per_page
        patch_in_page = patch_idx % self.patches_per_page
        
        # Get page dimensions
        page = self.pages[page_idx]
        w, h = page.size
        
        # Calculate patch position in grid
        grid_row = patch_in_page // self.grid_w
        grid_col = patch_in_page % self.grid_w
        
        # Calculate patch dimensions
        patch_h = h // self.grid_h
        patch_w = w // self.grid_w
        
        # Calculate bounding box
        x1 = grid_col * patch_w
        y1 = grid_row * patch_h
        x2 = min(x1 + patch_w, w)
        y2 = min(y1 + patch_h, h)
        
        return page_idx, x1, y1, x2, y2
    
    def highlight_patches(
        self,
        selected_indices: List[int],
        output_path: str,
        box_color: str = 'red',
        box_width: int = 5,
        opacity: float = 0.3,
        show_labels: bool = True,
        scores: Optional[List[float]] = None
    ) -> str:
        """
        Create highlighted PDF showing selected patches.
        
        Args:
            selected_indices: List of selected patch indices
            output_path: Path to save highlighted PDF
            box_color: Color for bounding boxes ('red', 'green', 'blue', etc.)
            box_width: Width of bounding box lines
            opacity: Opacity of highlight overlay (0-1)
            show_labels: Whether to show patch numbers
            scores: Optional relevance scores to display
        
        Returns:
            Path to saved highlighted PDF
        """
        print(f"\n🎨 Highlighting {len(selected_indices)} patches...")
        
        # Color mapping
        color_map = {
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'yellow': (255, 255, 0),
            'orange': (255, 165, 0),
            'purple': (128, 0, 128),
            'cyan': (0, 255, 255)
        }
        
        rgb_color = color_map.get(box_color.lower(), (255, 0, 0))
        
        # Group patches by page
        patches_by_page = {}
        for idx, patch_idx in enumerate(selected_indices):
            page_idx, x1, y1, x2, y2 = self._get_patch_bbox(patch_idx)
            
            if page_idx not in patches_by_page:
                patches_by_page[page_idx] = []
            
            patch_info = {
                'bbox': (x1, y1, x2, y2),
                'global_idx': patch_idx,
                'selection_order': idx + 1,
                'score': scores[idx] if scores else None
            }
            patches_by_page[page_idx].append(patch_info)
        
        # Create highlighted pages
        highlighted_pages = []
        
        for page_idx, page in enumerate(self.pages):
            # Convert to RGBA for transparency
            page_rgba = page.convert('RGBA')
            
            # Create overlay for highlights
            overlay = Image.new('RGBA', page_rgba.size, (255, 255, 255, 0))
            draw = ImageDraw.Draw(overlay)
            
            # Draw patches if this page has any
            if page_idx in patches_by_page:
                for patch_info in patches_by_page[page_idx]:
                    x1, y1, x2, y2 = patch_info['bbox']
                    
                    # Draw semi-transparent fill
                    fill_color = rgb_color + (int(255 * opacity),)
                    draw.rectangle([x1, y1, x2, y2], fill=fill_color, outline=None)
                    
                    # Draw solid border
                    border_color = rgb_color + (255,)
                    for i in range(box_width):
                        draw.rectangle(
                            [x1+i, y1+i, x2-i, y2-i],
                            outline=border_color,
                            width=1
                        )
                    
                    # Add label if requested
                    if show_labels:
                        label_text = f"#{patch_info['selection_order']}"
                        if patch_info['score'] is not None:
                            label_text += f"\n{patch_info['score']:.2f}"
                        
                        # Try to load font, fallback to default
                        try:
                            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
                        except:
                            font = ImageFont.load_default()
                        
                        # Draw label background
                        label_bbox = draw.textbbox((x1 + 5, y1 + 5), label_text, font=font)
                        draw.rectangle(
                            label_bbox,
                            fill=(255, 255, 255, 200)
                        )
                        
                        # Draw label text
                        draw.text(
                            (x1 + 5, y1 + 5),
                            label_text,
                            fill=border_color,
                            font=font
                        )
            
            # Composite overlay onto page
            highlighted = Image.alpha_composite(page_rgba, overlay)
            highlighted = highlighted.convert('RGB')
            highlighted_pages.append(highlighted)
        
        # Save as PDF
        if len(highlighted_pages) > 0:
            highlighted_pages[0].save(
                output_path,
                save_all=True,
                append_images=highlighted_pages[1:],
                resolution=self.dpi,
                quality=95
            )
            
            print(f"   ✓ Saved highlighted PDF: {output_path}")
            print(f"   • Pages with highlights: {len(patches_by_page)}")
            print(f"   • Total patches marked:  {len(selected_indices)}")
        
        return output_path
    
    def create_side_by_side_comparison(
        self,
        selected_indices: List[int],
        output_path: str,
        scores: Optional[List[float]] = None
    ) -> str:
        """
        Create side-by-side comparison: original vs highlighted.
        
        Args:
            selected_indices: List of selected patch indices
            output_path: Path to save comparison PDF
            scores: Optional relevance scores
        
        Returns:
            Path to saved comparison PDF
        """
        print(f"\n📊 Creating side-by-side comparison...")
        
        # Group patches by page
        patches_by_page = {}
        for idx, patch_idx in enumerate(selected_indices):
            page_idx, x1, y1, x2, y2 = self._get_patch_bbox(patch_idx)
            
            if page_idx not in patches_by_page:
                patches_by_page[page_idx] = []
            
            patches_by_page[page_idx].append({
                'bbox': (x1, y1, x2, y2),
                'selection_order': idx + 1,
                'score': scores[idx] if scores else None
            })
        
        comparison_pages = []
        
        for page_idx, page in enumerate(self.pages):
            if page_idx not in patches_by_page:
                continue  # Skip pages without selected patches
            
            w, h = page.size
            
            # Create comparison image (side by side)
            comparison = Image.new('RGB', (w * 2 + 20, h), (255, 255, 255))
            
            # Paste original on left
            comparison.paste(page, (0, 0))
            
            # Create highlighted version
            page_rgba = page.convert('RGBA')
            overlay = Image.new('RGBA', page_rgba.size, (255, 255, 255, 0))
            draw = ImageDraw.Draw(overlay)
            
            # Draw highlights
            for patch_info in patches_by_page[page_idx]:
                x1, y1, x2, y2 = patch_info['bbox']
                
                # Red semi-transparent fill
                draw.rectangle([x1, y1, x2, y2], fill=(255, 0, 0, 80), outline=None)
                
                # Solid red border
                for i in range(4):
                    draw.rectangle([x1+i, y1+i, x2-i, y2-i], outline=(255, 0, 0, 255), width=1)
                
                # Label
                label = f"#{patch_info['selection_order']}"
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
                except:
                    font = ImageFont.load_default()
                
                draw.text((x1 + 5, y1 + 5), label, fill=(255, 0, 0, 255), font=font)
            
            highlighted = Image.alpha_composite(page_rgba, overlay).convert('RGB')
            
            # Paste highlighted on right
            comparison.paste(highlighted, (w + 20, 0))
            
            # Add divider line
            draw_comp = ImageDraw.Draw(comparison)
            draw_comp.line([(w + 10, 0), (w + 10, h)], fill=(200, 200, 200), width=2)
            
            comparison_pages.append(comparison)
        
        # Save as PDF
        if len(comparison_pages) > 0:
            comparison_pages[0].save(
                output_path,
                save_all=True,
                append_images=comparison_pages[1:],
                resolution=self.dpi,
                quality=95
            )
            
            print(f"   ✓ Saved comparison: {output_path}")
        
        return output_path


def test_highlighter():
    """Test the highlighter with sample data."""
    import tempfile
    
    # This is a test function - requires actual PDF
    print("PDFHighlighter test requires a real PDF file.")
    print("Usage:")
    print("  highlighter = PDFHighlighter('document.pdf', patches_per_page=12)")
    print("  highlighter.highlight_patches([0, 5, 10, 15], 'highlighted.pdf')")


if __name__ == "__main__":
    test_highlighter()