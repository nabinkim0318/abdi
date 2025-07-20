import io

import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from bias_audit_tool.modeling.interpretation import generate_interpretation
from bias_audit_tool.stats.stats_analysis import run_anova
from bias_audit_tool.stats.stats_analysis import run_chi_square


def generate_pdf_report(df_proc, audit_cols, recommendations, group_col="race"):
    """
    Generate a PDF bias audit report including preprocessing suggestions,
    statistical interpretations, and visualizations.

    Args:
        df_proc (pd.DataFrame): The preprocessed DataFrame.
        audit_cols (list[str]): List of columns to audit for statistical
            differences and bias.
        recommendations (dict): Preprocessing strategies for each column.
        group_col (str, optional): Demographic grouping column used for comparison
            (e.g., 'race', 'gender'). Defaults to "race".

    Returns:
        io.BytesIO: A buffer containing the generated PDF report.

    Notes:
        - Preprocessing recommendations are listed per column.
        - Statistical tests:
            * Chi-square for categorical features (≤10 unique values)
            * ANOVA for continuous/numeric features
        - Distributions are visualized using histograms with KDE overlay.
        - Automatic pagination is handled when space is insufficient on
          the current page.
    """
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 50

    # Title
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, y, "📊 Bias Audit Report")
    y -= 30

    # Section 1: Preprocessing Recommendations
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "✅ Preprocessing Recommendations:")
    pdf.setFont("Helvetica", 10)
    for col, rec in recommendations.items():
        y -= 15
        pdf.drawString(60, y, f"- {col}: {rec}")
        if y < 100:
            pdf.showPage()
            y = height - 50

    # Section 2: Statistical Interpretation
    y -= 30
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "🧪 Statistical Interpretations:")
    pdf.setFont("Helvetica", 10)

    for col in audit_cols:
        try:
            if df_proc[col].nunique() <= 10:
                stat_result = run_chi_square(df_proc, col, None, group_col)
            else:
                stat_result = run_anova(df_proc, col, group_col)

            interpretation = generate_interpretation(stat_result)
            y -= 15
            pdf.drawString(60, y, f"- {interpretation}")
            if y < 100:
                pdf.showPage()
                y = height - 50

        except Exception as e:
            y -= 15
            pdf.drawString(60, y, f"- {col}: Error during interpretation: {e}")

    # Section 3: Visualizations
    y -= 30
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "📈 Feature Distributions:")
    for col in audit_cols:
        try:
            fig, ax = plt.subplots()
            sns.histplot(x=df_proc[col].dropna(), kde=True, ax=ax)
            imgdata = io.BytesIO()
            fig.savefig(imgdata, format="PNG", bbox_inches="tight")
            imgdata.seek(0)
            plt.close(fig)

            if y < 250:
                pdf.showPage()
                y = height - 50
            pdf.drawImage(ImageReader(imgdata), 50, y - 200, width=500, height=150)
            y -= 220

        except Exception as e:
            y -= 15
            pdf.setFont("Helvetica", 10)
            pdf.drawString(60, y, f"- Could not plot {col}: {e}")

    pdf.showPage()
    pdf.save()
    buffer.seek(0)
    return buffer
