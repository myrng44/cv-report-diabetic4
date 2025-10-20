import pdfplumber
import sys

# Set UTF-8 encoding for output
sys.stdout.reconfigure(encoding='utf-8')

pdf_path = r"C:\Users\ACER\PycharmProjects\Diabetic_CV_rp4\docs\s41598-025-93376-9.pdf"

with pdfplumber.open(pdf_path) as pdf:
    print(f"Total pages: {len(pdf.pages)}\n")
    for i, page in enumerate(pdf.pages):
        print(f"\n{'='*80}")
        print(f"PAGE {i+1}")
        print('='*80)
        text = page.extract_text()
        if text:
            print(text)
