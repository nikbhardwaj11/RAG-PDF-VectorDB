#!/usr/bin/env python3
"""
Download 10 public domain books from Project Gutenberg and convert to PDF.
Places PDFs in the books/ folder for the RAG system.
"""

import requests
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Project Gutenberg IDs for 10 diverse classic books
BOOKS = {
    "1342": "pride_and_prejudice",      # Pride and Prejudice - Jane Austen
    "345": "dracula",                    # Dracula - Bram Stoker
    "84": "frankenstein",                # Frankenstein - Mary Shelley
    "74": "tom_sawyer",                  # The Adventures of Tom Sawyer - Mark Twain
    "11": "alice_in_wonderland",         # Alice's Adventures in Wonderland - Lewis Carroll
    "64317": "great_gatsby",             # The Great Gatsby - F. Scott Fitzgerald
    "1260": "jane_eyre",                 # Jane Eyre - Charlotte Brontë
    "120": "treasure_island",            # Treasure Island - Robert Louis Stevenson
    "1184": "count_of_monte_cristo",     # The Count of Monte Cristo - Alexandre Dumas
    "36": "war_of_the_worlds",           # The War of the Worlds - H.G. Wells
}

BOOKS_DIR = Path(__file__).parent / "books"
BOOKS_DIR.mkdir(exist_ok=True)

def download_text(gutenberg_id: str, name: str) -> tuple[str, bool]:
    """Download plain text from Project Gutenberg."""
    url = f"https://www.gutenberg.org/cache/epub/{gutenberg_id}/pg{gutenberg_id}.txt"
    try:
        response = requests.get(url, timeout=30)
        if response.ok:
            # Clean up the text (remove Gutenberg header/footer)
            text = response.text
            # Find start of actual content (after "*** START OF")
            start_marker = "*** START OF"
            end_marker = "*** END OF"
            start_idx = text.find(start_marker)
            end_idx = text.find(end_marker)
            if start_idx != -1:
                start_idx = text.find("\n", start_idx) + 1
            if end_idx != -1:
                text = text[start_idx:end_idx]

            txt_path = BOOKS_DIR / f"{name}.txt"
            txt_path.write_text(text.strip(), encoding='utf-8')
            return (name, True)
        return (name, False)
    except Exception as e:
        print(f"  ✗ {name}: Download failed - {e}")
        return (name, False)

def convert_to_pdf(name: str) -> bool:
    """Convert .txt to .pdf using pandoc."""
    txt_path = BOOKS_DIR / f"{name}.txt"
    pdf_path = BOOKS_DIR / f"{name}.pdf"

    if not txt_path.exists():
        return False

    try:
        # Use pandoc with weasyprint for good PDF output
        result = subprocess.run([
            "pandoc",
            str(txt_path),
            "-o", str(pdf_path),
            "--pdf-engine=weasyprint",
            "-V", "geometry:margin=1in",
            "-V", "fontsize=11pt",
            "-V", "linestretch=1.3",
        ], capture_output=True, text=True, timeout=120)

        if result.returncode == 0 and pdf_path.exists():
            # Remove the .txt file after successful conversion
            txt_path.unlink()
            size_mb = pdf_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ {name}.pdf ({size_mb:.1f} MB)")
            return True
        else:
            print(f"  ✗ {name}: Pandoc failed - {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  ✗ {name}: Conversion timeout")
        return False
    except FileNotFoundError:
        print(f"  ✗ {name}: pandoc not installed")
        return False
    except Exception as e:
        print(f"  ✗ {name}: Conversion error - {e}")
        return False

def check_pandoc() -> bool:
    """Check if pandoc is available."""
    try:
        subprocess.run(["pandoc", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False

def main():
    print("=" * 60)
    print("Downloading 10 classic books from Project Gutenberg")
    print("=" * 60)

    # Check pandoc
    if not check_pandoc():
        print("\n❌ pandoc not found. Install it first:")
        print("   Windows: winget install pandoc")
        print("   Or: choco install pandoc")
        print("   Or download from: https://pandoc.org/installing.html")
        sys.exit(1)

    print("✓ pandoc found\n")

    # Download all texts in parallel
    print("📥 Downloading texts...")
    downloaded = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(download_text, gid, name): name
            for gid, name in BOOKS.items()
        }
        for future in as_completed(futures):
            name, success = future.result()
            if success:
                downloaded.append(name)
                print(f"  ✓ {name}.txt downloaded")
            else:
                print(f"  ✗ {name} download failed")

    if not downloaded:
        print("\n❌ No books downloaded. Check internet connection.")
        sys.exit(1)

    print(f"\n📄 Converting {len(downloaded)} books to PDF...")
    success_count = 0
    for name in downloaded:
        if convert_to_pdf(name):
            success_count += 1

    print("\n" + "=" * 60)
    print(f"✅ Complete! {success_count}/{len(BOOKS)} books ready in books/")
    print("=" * 60)

    # List final PDFs
    pdfs = list(BOOKS_DIR.glob("*.pdf"))
    print(f"\n📚 Total PDFs in books/: {len(pdfs)}")
    for pdf in sorted(pdfs):
        size_mb = pdf.stat().st_size / (1024 * 1024)
        print(f"   • {pdf.name} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()