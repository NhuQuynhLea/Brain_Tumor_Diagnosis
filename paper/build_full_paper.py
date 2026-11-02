import re
import pymupdf

# 1. Build full_paper_ieee.tex from main.tex
with open('main.tex', 'r', encoding='utf-8') as f:
    main_content = f.read()

def replacer(match):
    sec = match.group(1).strip()
    if not sec.endswith('.tex'):
        sec += '.tex'
    with open(sec, 'r', encoding='utf-8') as sf:
        return f"\n% --- Begin input {sec} ---\n" + sf.read() + f"\n% --- End input {sec} ---\n"

merged = re.sub(r'\\input\{([^}]+)\}', replacer, main_content)

with open('full_paper_ieee.tex', 'w', encoding='utf-8') as f:
    f.write(merged)

print(f"Successfully generated full_paper_ieee.tex, length: {len(merged)}")

# 2. Sync full_paper_ieee.pdf
import shutil
shutil.copyfile('main.pdf', 'full_paper_ieee.pdf')
print("Successfully synced main.pdf -> full_paper_ieee.pdf")

# 3. Render PDF pages to PNG
doc = pymupdf.open('main.pdf')
for i, page in enumerate(doc):
    pix = page.get_pixmap(dpi=150)
    pix.save(f'page_{i+1}.png')
print(f"Successfully rendered {len(doc)} pages to PNG.")
