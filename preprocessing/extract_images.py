import sys
import argparse  # for terminal inputs
import hashlib
from pathlib import Path  # for handling file paths
import shutil
import subprocess
import re
import json
import pymupdf4llm                 
import pymupdf as fitz          
from PIL import Image, UnidentifiedImageError

# How small are images allowed to be
MIN_IMAGE_WIDTH = 40
MIN_IMAGE_HEIGHT = 40
MIN_IMAGE_AREA = 2500


# Supported file extensions
SUPPORTED_EXTENSIONS = {".docx", ".odt", ".pptx", ".pdf"}

# Regular expression pattern for catching both markdown and HTML images
IMAGE_PATTERN = re.compile(
    r'!\[(?P<alt>[^\]]*)\]\((?P<src1>[^)\s]+)[^)]*\)'
    r'|<img\b[^>]*?\bsrc="(?P<src2>[^"]+)"[^>]*>'
)


# Giving each document a unique identifier with hashlib
def get_document_id(relative_path: str) -> str:
    path = relative_path.encode('utf-8')
    path_hash = hashlib.sha1(path)
    short_hash = path_hash.hexdigest()[:8]  # First 8 hash characters as ID
    return short_hash  # Fixed: returns short_hash instead of unassigned 'hash'


# Uses pandoc to convert a Word/Office document to markdown and extract media
def convert_to_markdown(source_file: Path, media_dir: Path) -> str:
    media_dir.mkdir(parents=True, exist_ok=True)
    
    pandoc_bin = shutil.which("pandoc")
    if not pandoc_bin:
        try:
            import pypandoc
            pandoc_bin = pypandoc.get_pandoc_path()
        except Exception:
            raise RuntimeError("Pandoc mangler (pip install pypandoc_binary)")

    result = subprocess.run(
        [pandoc_bin, str(source_file), "-t", "gfm", "--wrap=none", f"--extract-media={media_dir}"], 
        capture_output=True, text=True, timeout=120
    )

    if result.returncode != 0:
        raise RuntimeError(f"Pandoc conversion failed: {result.stderr[:200]}")

    return result.stdout


def pdf_to_md_plain(source_file: Path, media_dir: Path) -> str:
    #Reserve: pure text and images without order
    media_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(source_file)
    markdown_lines = []
    img_counter = 1
    for page in doc:
        markdown_lines.append(page.get_text())
        for img_info in page.get_images(full=True):
            base_image = doc.extract_image(img_info[0])
            img_filename = f"image_{img_counter:03d}.{base_image['ext']}"
            (media_dir / img_filename).write_bytes(base_image["image"])
            markdown_lines.append(f"\n\n![embedded image]({img_filename})\n\n")
            img_counter += 1
    return "\n".join(markdown_lines)


def pdf_to_md(source_file: Path, media_dir: Path) -> str:
    # PDF to markdown WITH headlines - goes back to without if fail
    media_dir.mkdir(parents=True, exist_ok=True)
    tmp_pdf = media_dir / "doc.pdf" 
    shutil.copy2(source_file, tmp_pdf)
    try:
        return pymupdf4llm.to_markdown(
            str(tmp_pdf), write_images=True,
            image_path=str(media_dir), image_format="png")
    except Exception as e:
        print(f"  pymupdf4llm failed ({type(e).__name__}: {str(e)[:100]}), uses only text")
        return pdf_to_md_plain(source_file, media_dir)


def should_keep_image(image_path: Path) -> bool:
    #Check if an image is large enough to be useful.

    try:
        with Image.open(image_path) as img:
            width, height = img.size

            # Reject images that are too narrow or too short
            if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                return False

            # Reject images with too few total pixels
            if width * height < MIN_IMAGE_AREA:
                return False

            return True

    except (UnidentifiedImageError, OSError) as e:
        print(f"  Warning: Could not inspect image {image_path}: {e}")

        # Keep unknown formats rather than accidentally losing useful images
        return True


# Image extraction and text replacement with regex
def process_images_and_text(
    markdown_text: str,
    media_dir: Path,
    doc_id: str,
    out_root: Path,
    rel_path: str
):
    image_records = []
    skipped_images = 0

    def replace_image(match):
        nonlocal skipped_images

        src = match.group("src1") or match.group("src2")

        # Locate the extracted image
        img_path = Path(src)

        if not img_path.is_file():
            hits = list(media_dir.rglob(img_path.name))
            img_path = hits[0] if hits else None

        # Filter out small images before creating records
        if img_path and img_path.is_file():
            if not should_keep_image(img_path):
                skipped_images += 1
                return ""

        # Assign IDs only to retained images
        idx = len(image_records) + 1
        img_id = f"{doc_id}_{idx:03d}"

        record = {
            "id": img_id,
            "source_doc": rel_path,
            "order": idx,
            "file_path": None
        }

        if img_path and img_path.is_file():
            dest_dir = out_root / "_images" / doc_id
            dest_dir.mkdir(parents=True, exist_ok=True)

            dest_file = dest_dir / f"img_{idx:03d}{img_path.suffix.lower()}"

            shutil.copy2(img_path, dest_file)

            record["file_path"] = dest_file.relative_to(
                out_root
            ).as_posix()

        image_records.append(record)

        return f"\n\n[image: {img_id}]\n\n"

    cleaned_markdown = IMAGE_PATTERN.sub(
        replace_image,
        markdown_text
    )

    if skipped_images:
        print(f"  Filtered out {skipped_images} small images")

    return cleaned_markdown, image_records



def clean_markdown_text(text: str) -> str:
    #Performs light cleanups on the markdown text to make it cleaner to read.
    text = text.replace("\u00a0", " ")  # Non-breaking spaces to normal spaces
    text = re.sub(r"</?[u]>", "", text) # Remove underline tags
    text = re.sub(r"\n{3,}", "\n\n", text) # Max one blank line
    text = re.sub(r"^(#+)\s*\*\*(.*?)\*\*\s*$", r"\1 \2", text, flags=re.M)
    return text.strip() + "\n"


def process_document(file_path: Path, src_root: Path, out_root: Path):
    #Handles the full workflow for a single document.
    rel_path = file_path.relative_to(src_root).as_posix()
    doc_id = get_document_id(rel_path)
    temp_media_dir = out_root / "_tmp" / doc_id
    
    doc_record = {"source": rel_path, "doc_id": doc_id, "status": "ok"}
    images = []
    
    try:
        # 1. Convert to markdown (uses PDF extractor if it's a PDF, otherwise Pandoc)
        if file_path.suffix.lower() == ".pdf":
            raw_md = pdf_to_md(file_path, temp_media_dir)
        else:
            raw_md = convert_to_markdown(file_path, temp_media_dir)
        
        # 2. Extract/relocate images and inject custom markers
        marked_md, images = process_images_and_text(raw_md, temp_media_dir, doc_id, out_root, rel_path)
        
        # 3. Clean up the text
        final_md = clean_markdown_text(marked_md)
        
        # 4. Save the resulting Markdown file
        out_file = out_root / (rel_path + ".md")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(final_md, encoding="utf-8")
        
        doc_record["output_file"] = out_file.relative_to(out_root).as_posix()
        doc_record["image_count"] = len(images)
        doc_record["chars"] = len(final_md)
        
    except Exception as e:
        doc_record["status"] = "error"
        doc_record["error"] = str(e)
    finally:
        shutil.rmtree(temp_media_dir, ignore_errors=True)
        
    return doc_record, images


def main():
    # Positional terminal arguments: python3 script.py <source> <output>
    parser = argparse.ArgumentParser(description="Convert documents to Markdown with extracted images.")
    parser.add_argument("source", help="Path to the source folder containing documents")
    parser.add_argument("output", help="Path to the output destination folder")
    parsed_args = parser.parse_args()

    src_root = Path(parsed_args.source).resolve()
    out_root = Path(parsed_args.output).resolve()

    if not src_root.is_dir():
        sys.exit(f"Error: Source directory does not exist: {src_root}")
    if out_root == src_root or src_root in out_root.parents:
        sys.exit("Error: Output directory cannot be inside the source directory.")

    out_root.mkdir(parents=True, exist_ok=True)

    # Collect all target files recursively using our SUPPORTED_EXTENSIONS
    EXCLUDED_TERMS = ("sensitiv", "fortrolig", "confidential")

    files_to_process = [
        p for p in sorted(src_root.rglob("*"))
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_EXTENSIONS
        and not p.name.startswith("~$")
        and not any(
            term in p.relative_to(src_root).as_posix().lower()
            for term in EXCLUDED_TERMS
        )
    ]

    print(f"Found {len(files_to_process)} documents to process.")

    all_docs = []
    all_images = []

    for i, file_path in enumerate(files_to_process, 1):
        rel_path = file_path.relative_to(src_root).as_posix()
        print(f"[{i}/{len(files_to_process)}] Processing: {rel_path}")
        
        doc_rec, img_recs = process_document(file_path, src_root, out_root)
        all_docs.append(doc_rec)
        all_images.extend(img_recs)

    # Write JSONL summaries
    with open(out_root / "index.jsonl", "w", encoding="utf-8") as f:
        for doc in all_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    with open(out_root / "images.jsonl", "w", encoding="utf-8") as f:
        for img in all_images:
            f.write(json.dumps(img, ensure_ascii=False) + "\n")

    # Final cleanup of global temp folder if it exists
    shutil.rmtree(out_root / "_tmp", ignore_errors=True)
    
    print("\nProcessing complete!")
    print(f"Results saved to: {out_root}")


if __name__ == "__main__":
    main()