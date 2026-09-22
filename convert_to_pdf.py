from fpdf import FPDF
import os

for fname in sorted(os.listdir("eval/resumes")):
    if not fname.endswith(".txt"):
        continue
    stem = fname.replace(".txt", "")
    txt_path = f"eval/resumes/{fname}"
    pdf_path = f"eval/resumes/{stem}.pdf"

    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    pdf.set_auto_page_break(auto=True, margin=15)
    for line in text.split("\n"):
        pdf.cell(0, 5, line.encode("latin-1", "replace").decode("latin-1"), ln=True)
    pdf.output(pdf_path)
    os.remove(txt_path)
    print(f"converted {fname} → {stem}.pdf")

print("Done.")