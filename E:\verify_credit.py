import sys, os, time, win32com.client as win32
sys.stdout.reconfigure(encoding='utf-8')

DOCX = r'E:\hermes_playground\paper-writing\projects\dfc-paper\DFC_Liu_Jun3_2026_revision.docx'
PDF  = r'E:\hermes_playground\paper-writing\projects\dfc-paper\tmp\dfc_after_credit_rewrite.pdf'

os.system('taskkill /F /IM WINWORD.EXE 2>nul >nul')
time.sleep(1)
word = win32.gencache.EnsureDispatch('Word.Application')
word.Visible = False
word.DisplayAlerts = 0

try:
    doc = word.Documents.Open(DOCX)
    print(f"OK Word opened. Total paragraphs: {doc.Paragraphs.Count}")

    found = False
    for i, p in enumerate(doc.Paragraphs):
        text = p.Range.Text.strip()
        if 'CRediT authorship contribution statement' in text or 'CRediT Author Contributions' in text:
            found = True
            print(f"\n=== CRediT section at Word paragraph {i} ===")
            for j in range(i, min(i+10, doc.Paragraphs.Count)):
                t = doc.Paragraphs[j+1].Range.Text.strip()
                sname = doc.Paragraphs[j+1].Style.NameLocal
                print(f"  [{j}] style={sname!r}: {t[:160]}")
            break
    if not found:
        print("NOT FOUND")

    doc.ExportAsFixedFormat(
        OutputFileName=PDF, ExportFormat=17,
        OpenAfterExport=False, OptimizeFor=0, Range=0, Item=0,
        IncludeDocProps=True, KeepIRM=True, CreateBookmarks=0,
        DocStructureTags=True, BitmapMissingFonts=True, UseISO19005_1=False
    )
    print(f"\nPDF -> {PDF} ({os.path.getsize(PDF):,} bytes)")
finally:
    try: doc.Close(False)
    except: pass
    word.Quit()
    time.sleep(0.5)
    os.system('taskkill /F /IM WINWORD.EXE 2>nul >nul')