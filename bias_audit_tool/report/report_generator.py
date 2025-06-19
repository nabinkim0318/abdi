import io

import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


def generate_pdf_report(df_proc, audit_cols, recommendations):
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 50

    pdf.drawString(50, y, "📊 Bias Audit Report")
    y -= 30

    pdf.drawString(50, y, "✅ Preprocessing Recommendations:")
    for col, rec in recommendations.items():
        y -= 15
        pdf.drawString(60, y, f"- {col}: {rec}")

    y -= 30
    pdf.drawString(50, y, "📈 Visualizations:")
    for col in audit_cols:
        fig, ax = plt.subplots()
        sns.histplot(x=df_proc[col].dropna(), kde=True, ax=ax)
        imgdata = io.BytesIO()
        fig.savefig(imgdata, format="PNG")
        imgdata.seek(0)
        plt.close(fig)
        if y < 250:
            pdf.showPage()
            y = height - 50
        pdf.drawImage(ImageReader(imgdata), 50, y - 200, width=500, height=150)
        y -= 220

    pdf.showPage()
    pdf.save()
    buffer.seek(0)
    return buffer
