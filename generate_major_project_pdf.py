"""
================================================================================
MAJOR PROJECT DOCUMENTATION GENERATOR
Compiles full project documentation into a publication-grade PDF using ReportLab
================================================================================
"""

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

# Palette constants matching DiabetesCare AI (Sage Green Scandinavian Theme)
COLOR_PRIMARY = colors.HexColor("#557D6C")      # Sage Green
COLOR_DARK = colors.HexColor("#345747")         # Dark Green
COLOR_DEEP = colors.HexColor("#28483A")         # Deep Green
COLOR_MINT = colors.HexColor("#E5F0E9")         # Mint Background
COLOR_MINT2 = colors.HexColor("#F0F6F1")        # Soft Mint Light
COLOR_BG = colors.HexColor("#F5F8F5")           # Canvas Background
COLOR_INK = colors.HexColor("#263B32")          # Forest Ink Text
COLOR_MUTED = colors.HexColor("#7A8B81")        # Muted Text
COLOR_LINE = colors.HexColor("#E2ECE4")         # Border Lines
COLOR_ROSE = colors.HexColor("#C77C68")         # Critical Alert Rose
COLOR_AMBER = colors.HexColor("#C58A65")        # Warning Amber
COLOR_WHITE = colors.white

class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and print total page count."""
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_decorations(self, total_pages):
        # Suppress headers/footers on cover page
        if self._pageNumber == 1:
            return

        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(COLOR_MUTED)

        # Running Header
        self.drawString(54, letter[1] - 36, "AuraHealth AI • Major Project: Diabetes Risk Prediction Using Machine Learning")
        self.drawRightString(letter[0] - 54, letter[1] - 36, "Clinical Decision Support System (CDSS)")
        self.setStrokeColor(COLOR_LINE)
        self.setLineWidth(0.75)
        self.line(54, letter[1] - 42, letter[0] - 54, letter[1] - 42)

        # Running Footer
        self.line(54, 46, letter[0] - 54, 46)
        self.drawString(54, 34, "Confidential • Student Project Specification & Clinical Triage Benchmark")
        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(letter[0] - 54, 34, page_str)
        self.restoreState()


def build_pdf_documentation(output_pdf_path: str):
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom Typography Styles
    style_cover_title = ParagraphStyle(
        'CoverTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=30,
        textColor=COLOR_DEEP,
        alignment=0,
        spaceAfter=10
    )

    style_cover_sub = ParagraphStyle(
        'CoverSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=17,
        textColor=COLOR_PRIMARY,
        spaceAfter=25
    )

    style_meta_label = ParagraphStyle(
        'MetaLabel',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=13,
        textColor=COLOR_DARK
    )

    style_meta_val = ParagraphStyle(
        'MetaVal',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=COLOR_INK
    )

    style_h1 = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=19,
        textColor=COLOR_DEEP,
        spaceBefore=16,
        spaceAfter=8,
        keepWithNext=True
    )

    style_h2 = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=COLOR_DARK,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )

    style_body = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=14,
        textColor=COLOR_INK,
        spaceAfter=8
    )

    style_body_bold = ParagraphStyle(
        'Body_Bold_Custom',
        parent=style_body,
        fontName='Helvetica-Bold'
    )

    style_bullet = ParagraphStyle(
        'Bullet_Custom',
        parent=style_body,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )

    style_code = ParagraphStyle(
        'Code_Custom',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8,
        leading=11,
        textColor=COLOR_DEEP
    )

    style_callout = ParagraphStyle(
        'CalloutText',
        parent=style_body,
        textColor=COLOR_DARK,
        fontSize=8.5,
        leading=13
    )

    style_th = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=COLOR_WHITE,
        alignment=0
    )

    style_td = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=COLOR_INK
    )

    style_td_mono = ParagraphStyle(
        'TableCellMono',
        parent=style_td,
        fontName='Courier-Bold',
        fontSize=8
    )

    story = []

    # ==============================================================================
    # COVER PAGE / TITLE BLOCK
    # ==============================================================================
    story.append(Spacer(1, 20))
    story.append(Paragraph("MAJOR PROJECT SPECIFICATION & TECHNICAL DOSSIER", ParagraphStyle('Eyebrow', fontName='Helvetica-Bold', fontSize=10, leading=12, textColor=COLOR_PRIMARY, spaceAfter=8)))
    story.append(Paragraph("Diabetes Risk Prediction Using Machine Learning", style_cover_title))
    story.append(Paragraph("A Capacity-Constrained Clinical Decision Support System (CDSS) for Rural Health Clinics with 30% Diagnostic Lab Testing Ceilings", style_cover_sub))
    story.append(HRFlowable(width="100%", thickness=2, color=COLOR_PRIMARY, spaceAfter=20, spaceBefore=0))

    # Metadata Card Table
    meta_data = [
        [Paragraph("Candidate Name:", style_meta_label), Paragraph("Yashraj", style_meta_val), Paragraph("Target Domain:", style_meta_label), Paragraph("Clinical Machine Learning / Healthcare Analytics", style_meta_val)],
        [Paragraph("Project Track:", style_meta_label), Paragraph("Major Project (End-to-End CDSS)", style_meta_val), Paragraph("Target Dataset:", style_meta_label), Paragraph("Pima Indians Diabetes Database (N=768)", style_meta_val)],
        [Paragraph("Deliverables:", style_meta_label), Paragraph("Console Pipeline, Web GUI, Technical PDF", style_meta_val), Paragraph("Core Constraint:", style_meta_label), Paragraph("Max 30% Daily Walk-in Lab Testing Capacity", style_meta_val)],
        [Paragraph("Winning Model:", style_meta_label), Paragraph("Distance-Weighted KNN (k=9)", style_meta_val), Paragraph("Primary Metric:", style_meta_label), Paragraph("Recall @ 30% Capacity (57.4% Captured)", style_meta_val)]
    ]
    t_meta = Table(meta_data, colWidths=[90, 160, 95, 155])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), COLOR_MINT2),
        ('BOX', (0,0), (-1,-1), 1, COLOR_LINE),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 22))

    # Executive Summary Callout
    exec_text = (
        "<b>Executive Summary:</b> In rural primary health centers, laboratory equipment, chemical reagents, and certified phlebotomy "
        "personnel are severely constrained. This major project designs, evaluates, and deploys a clinical screening tool that "
        "ranks daily walk-in arrivals by predictive risk to identify the top 30% highest-risk patients for full venous blood draws "
        "(Fasting Plasma Glucose / HbA1c). By engineering a zero-leakage scikit-learn pipeline with median imputation for hidden zeros, "
        "feature scaling, and dynamic capacity-bound threshold tuning, the system maximizes diabetic patient capture while keeping "
        "false alarms to a minimum."
    )
    t_exec = Table([[Paragraph(exec_text, style_callout)]], colWidths=[500])
    t_exec.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), COLOR_MINT),
        ('BOX', (0,0), (-1,-1), 1, COLOR_PRIMARY),
        ('PADDING', (0,0), (-1,-1), 12),
        ('LEFTPADDING', (0,0), (-1,-1), 14),
    ]))
    story.append(t_exec)
    story.append(Spacer(1, 20))

    # ==============================================================================
    # 1. BUSINESS PROBLEM & 30% CAPACITY FORMULATION
    # ==============================================================================
    story.append(Paragraph("1. Business Problem & Mathematical Capacity Formulation", style_h1))
    story.append(Paragraph(
        "A rural clinic receives a variable daily volume of walk-in patients presenting vague metabolic or lifestyle risk factors. "
        "Due to limited laboratory reagents and centrifuge equipment, the on-site lab can afford to run diagnostic blood tests for "
        "<b>at most 30% of daily arrivals</b>. The remaining 70% must be triaged home with non-invasive lifestyle guidance.",
        style_body
    ))
    story.append(Paragraph(
        "<b>The Clinical Loss Asymmetry:</b> In clinical medicine, the penalty of a False Negative (FN) drastically exceeds that of a False Positive (FP):",
        style_body
    ))
    story.append(Paragraph("• <b>Cost of False Negative (C_FN &gt;&gt; C_FP):</b> A diabetic patient is misclassified as healthy and sent home without testing. Untreated hyperglycemia progresses silently into irreversible microvascular pathologies: diabetic retinopathy (blindness), nephropathy (kidney failure), peripheral neuropathy, and acute ketoacidosis.", style_bullet))
    story.append(Paragraph("• <b>Cost of False Positive (C_FP):</b> A healthy patient is flagged as high-risk and receives a blood draw. The test returns normal. The clinical cost is confined to the loss of 1 test kit and 15 minutes of lab technician time.", style_bullet))
    story.append(Paragraph(
        "<b>Mathematical Formulation:</b> Let <i>N</i> be the daily cohort size. The clinic testing quota is <i>K = ceil(0.30 &times; N)</i>. "
        "Rather than applying a generic 0.50 cutoff, the classifier estimates continuous risk probabilities <i>p_hat = P(Diabetic | X)</i>, "
        "sorts arrivals descending, and assigns the top <i>K</i> patients to the phlebotomy queue. The target optimization metric is "
        "<b>Recall@30%</b>: the percentage of all actual diabetics captured within the top 30% screened cohort.",
        style_body
    ))
    story.append(Spacer(1, 10))

    # ==============================================================================
    # 2. DATASET OVERVIEW & PHYSIOLOGICAL MEANING
    # ==============================================================================
    story.append(Paragraph("2. Dataset Overview & Physiological Feature Meanings", style_h1))
    story.append(Paragraph(
        "The project is evaluated on the <b>Pima Indians Diabetes Database</b> (National Institute of Diabetes and Digestive and Kidney Diseases). "
        "The cohort consists of <b>768 female patients</b> of Pima heritage aged &ge;21 years, containing 8 physiological predictor variables and 1 binary outcome.",
        style_body
    ))

    feat_table_data = [
        [Paragraph("Feature", style_th), Paragraph("Type", style_th), Paragraph("Clinical Range", style_th), Paragraph("Hidden Zeros?", style_th), Paragraph("Physiological Description", style_th)],
        [Paragraph("Pregnancies", style_td_mono), Paragraph("Integer", style_td), Paragraph("0 - 17", style_td), Paragraph("Valid (14.5%)", style_td), Paragraph("Gestational count. Nulliparous women have 0 pregnancies. Valid.", style_td)],
        [Paragraph("Glucose", style_td_mono), Paragraph("Numeric", style_td), Paragraph("0 - 199 mg/dL", style_td), Paragraph("5 (0.7%)", style_td), Paragraph("2-hr oral glucose tolerance test. Glucose &lt; 40 causes coma; 0 is missing.", style_td)],
        [Paragraph("BloodPressure", style_td_mono), Paragraph("Numeric", style_td), Paragraph("0 - 122 mmHg", style_td), Paragraph("35 (4.6%)", style_td), Paragraph("Diastolic BP. 0 indicates cardiac arrest; represents omitted measurement.", style_td)],
        [Paragraph("SkinThickness", style_td_mono), Paragraph("Numeric", style_td), Paragraph("0 - 99 mm", style_td), Paragraph("227 (29.6%)", style_td), Paragraph("Triceps skinfold caliper measurement of subcutaneous fat layer.", style_td)],
        [Paragraph("Insulin", style_td_mono), Paragraph("Numeric", style_td), Paragraph("0 - 846 muU/mL", style_td), Paragraph("374 (48.7%)", style_td), Paragraph("2-hr post-prandial serum insulin. Skipped by 48.7% of clinic walk-ins.", style_td)],
        [Paragraph("BMI", style_td_mono), Paragraph("Numeric", style_td), Paragraph("0 - 67.1 kg/m2", style_td), Paragraph("11 (1.4%)", style_td), Paragraph("Body mass index (weight / height^2). Zero indicates missing value.", style_td)],
        [Paragraph("DPF", style_td_mono), Paragraph("Numeric", style_td), Paragraph("0.078 - 2.42", style_td), Paragraph("None (0.0%)", style_td), Paragraph("Diabetes Pedigree Function: hereditary family history genetic score.", style_td)],
        [Paragraph("Age", style_td_mono), Paragraph("Integer", style_td), Paragraph("21 - 81 yrs", style_td), Paragraph("None (0.0%)", style_td), Paragraph("Patient age in years. Adult population only.", style_td)],
        [Paragraph("Outcome", style_td_mono), Paragraph("Target", style_td), Paragraph("0 or 1", style_td), Paragraph("Target", style_td), Paragraph("0 = Non-diabetic (65.1%, N=500), 1 = Diabetic (34.9%, N=268).", style_td)]
    ]
    t_feat = Table(feat_table_data, colWidths=[75, 45, 75, 75, 230])
    t_feat.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), COLOR_DARK),
        ('GRID', (0,0), (-1,-1), 0.5, COLOR_LINE),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [COLOR_WHITE, COLOR_MINT2]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(t_feat)
    story.append(Spacer(1, 14))

    # ==============================================================================
    # 3. HIDDEN ZERO AUDITING & DATA LEAKAGE SHIELD
    # ==============================================================================
    story.append(Paragraph("3. Hidden Zero Treatment & Data Leakage Shield", style_h1))
    story.append(Paragraph(
        "<b>The Hidden Zero Phenomenon:</b> When compiling the Pima dataset in the 1990s, missing laboratory tests were recorded as numerical "
        "zeros rather than explicit null tokens. Standard software calls (e.g. <code>df.isnull().sum()</code>) report zero missing values, creating "
        "a dangerous trap for novice engineers who treat zero insulin or zero blood pressure as genuine biological measurements.",
        style_body
    ))
    story.append(Paragraph(
        "<b>The Catastrophe of Deleting Rows:</b> If rows containing zeros for Insulin (374) or SkinThickness (227) are deleted, "
        "<b>376 out of 768 rows (over 50% of the dataset)</b> are lost. This destroys statistical sample size and introduces severe "
        "demographic selection bias. Therefore, all 768 records are preserved.",
        style_body
    ))
    story.append(Paragraph(
        "<b>The Strict Data Leakage Boundary:</b> Computing replacement medians across the entire 768 rows <i>before</i> train/test partitioning "
        "constitutes examination-failing data leakage. Future test patient distributions would contaminate historical training medians. "
        "To guarantee clinical integrity, we employ an encapsulated <code>sklearn.pipeline.Pipeline</code>: invalid zeros are converted to <code>np.nan</code>, "
        "and <code>SimpleImputer(strategy='median')</code> is fitted <b>strictly on the training split (614 instances)</b> and transformed onto test data.",
        style_body
    ))
    story.append(Spacer(1, 10))

    # ==============================================================================
    # 4. DATA PARTITIONING & FEATURE SCALING PIPELINE
    # ==============================================================================
    story.append(Paragraph("4. Stratified Partitioning & Feature Scaling Pipeline", style_h1))
    story.append(Paragraph(
        "<b>Stratified 80/20 Splitting:</b> The dataset has a 65.1% negative to 34.9% positive class imbalance. A naive random split risks "
        "unbalanced positive patient distributions. Using <code>train_test_split(..., stratify=y, test_size=0.20, random_state=42)</code> "
        "enforces exactly 34.9% diabetic prevalence in both the training set (614 rows: 214 diabetic, 400 healthy) and the holdout test set "
        "(154 rows: 54 diabetic, 100 healthy).",
        style_body
    ))
    story.append(Paragraph(
        "<b>StandardScaler Normalization:</b> Continuous features are normalized to zero mean and unit variance (<i>z = (x - mu) / sigma</i>). "
        "This is strictly mandatory for <b>K-Nearest Neighbors</b> (where unscaled insulin ranging 0-846 completely drowns out pedigree scores ranging 0.08-2.42) "
        "and <b>Logistic Regression</b> (where L2 weight penalization requires uniform feature scales). Decision Trees are scale-invariant but benefit "
        "from uniform pipeline interface packaging.",
        style_body
    ))
    story.append(Spacer(1, 10))

    # ==============================================================================
    # 5. EXPERIMENTAL BENCHMARK RESULTS
    # ==============================================================================
    story.append(Paragraph("5. Experimental Benchmark Results & Model Selection", style_h1))
    story.append(Paragraph(
        "Three distinct supervised learning paradigms were benchmarked: <b>K-Nearest Neighbors (k=9, distance-weighted)</b>, "
        "<b>Logistic Regression (L2 penalty, C=1.0)</b>, and <b>Decision Tree (CART, max_depth=4, min_samples_leaf=10)</b>. "
        "Models were trained with 5-fold stratified cross-validation on the training set and evaluated on the holdout test cohort (N=154).",
        style_body
    ))

    # Table: Default Threshold Metrics
    story.append(Paragraph("<b>Table 1: Performance at Standard Threshold (T = 0.50)</b>", style_h2))
    t1_data = [
        [Paragraph("Model Candidate", style_th), Paragraph("Accuracy", style_th), Paragraph("Precision", style_th), Paragraph("Recall (Sens.)", style_th), Paragraph("F1-Score", style_th), Paragraph("ROC-AUC", style_th), Paragraph("5-Fold CV AUC", style_th)],
        [Paragraph("KNN (k=9, distance)", style_td_mono), Paragraph("74.0%", style_td), Paragraph("63.5%", style_td), Paragraph("61.1%", style_td), Paragraph("0.623", style_td), Paragraph("0.797", style_td), Paragraph("0.817", style_td)],
        [Paragraph("Logistic Regression", style_td_mono), Paragraph("70.8%", style_td), Paragraph("60.0%", style_td), Paragraph("50.0%", style_td), Paragraph("0.546", style_td), Paragraph("0.813", style_td), Paragraph("0.843", style_td)],
        [Paragraph("Decision Tree (depth=4)", style_td_mono), Paragraph("79.2%", style_td), Paragraph("69.6%", style_td), Paragraph("72.2%", style_td), Paragraph("0.709", style_td), Paragraph("0.801", style_td), Paragraph("0.778", style_td)]
    ]
    t_m1 = Table(t1_data, colWidths=[120, 60, 60, 70, 60, 65, 65])
    t_m1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), COLOR_DARK),
        ('GRID', (0,0), (-1,-1), 0.5, COLOR_LINE),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [COLOR_WHITE, COLOR_MINT2]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_m1)
    story.append(Spacer(1, 10))

    # Table: 30% Capacity Constrained Evaluation
    story.append(Paragraph("<b>Table 2: Clinical Performance Under 30% Testing Capacity Quota (47 Tests / 154 Patients)</b>", style_h2))
    t2_data = [
        [Paragraph("Model Candidate", style_th), Paragraph("Tests Allowed", style_th), Paragraph("Caught (TP)", style_th), Paragraph("Missed (FN)", style_th), Paragraph("Wasted (FP)", style_th), Paragraph("Recall @ 30%", style_th), Paragraph("Precision @ 30%", style_th), Paragraph("Operating Cutoff", style_th)],
        [Paragraph("KNN (Winner)", style_td_mono), Paragraph("47", style_td), Paragraph("31", style_td), Paragraph("23", style_td), Paragraph("16", style_td), Paragraph("57.4%", style_td_mono), Paragraph("66.0%", style_td), Paragraph("p >= 0.536", style_td_mono)],
        [Paragraph("Decision Tree", style_td_mono), Paragraph("47", style_td), Paragraph("30", style_td), Paragraph("24", style_td), Paragraph("17", style_td), Paragraph("55.6%", style_td), Paragraph("63.8%", style_td), Paragraph("p >= 0.557", style_td)],
        [Paragraph("Logistic Regression", style_td_mono), Paragraph("47", style_td), Paragraph("27", style_td), Paragraph("27", style_td), Paragraph("20", style_td), Paragraph("50.0%", style_td), Paragraph("57.4%", style_td), Paragraph("p >= 0.499", style_td)]
    ]
    t_m2 = Table(t2_data, colWidths=[110, 55, 55, 55, 55, 65, 65, 60])
    t_m2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), COLOR_DARK),
        ('GRID', (0,0), (-1,-1), 0.5, COLOR_LINE),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [COLOR_MINT, COLOR_WHITE, COLOR_WHITE]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_m2)
    story.append(Spacer(1, 10))

    # Embedded Confusion Matrix Figure
    cm_img_path = os.path.join("outputs", "confusion_matrices", "all_models_confusion_matrices.png")
    if os.path.exists(cm_img_path):
        story.append(Spacer(1, 6))
        story.append(Paragraph("<b>Figure 1: Comparative Confusion Matrices across KNN, Logistic Regression, and Decision Tree</b>", style_body_bold))
        story.append(Image(cm_img_path, width=480, height=145))
        story.append(Spacer(1, 10))

    # ==============================================================================
    # 6. MONDAY MORNING CASE STUDY
    # ==============================================================================
    story.append(Paragraph("6. Monday Morning Walk-in Triage Case Study (Phase 5 Validation)", style_h1))
    story.append(Paragraph(
        "To validate the system in operational conditions, a simulated cohort of <b>20 walk-in patients</b> arrived at the clinic. "
        "With a strict 30% quota, the lab permitted <b>exactly 6 diagnostic blood tests</b>. Patients were scored and ranked descending:",
        style_body
    ))

    queue_data = [
        [Paragraph("Rank", style_th), Paragraph("Patient ID", style_th), Paragraph("Risk Score", style_th), Paragraph("Glucose", style_th), Paragraph("BMI", style_th), Paragraph("Age", style_th), Paragraph("Assigned Clinical Action", style_th)],
        [Paragraph("#1", style_td_mono), Paragraph("P-112", style_td_mono), Paragraph("90.1%", style_td_mono), Paragraph("168 mg/dL", style_td), Paragraph("38.0", style_td), Paragraph("34", style_td), Paragraph("PRIORITY LAB BLOOD TEST (Slot 1)", style_td_mono)],
        [Paragraph("#2", style_td_mono), Paragraph("P-105", style_td_mono), Paragraph("88.9%", style_td_mono), Paragraph("137 mg/dL", style_td), Paragraph("43.1", style_td), Paragraph("33", style_td), Paragraph("PRIORITY LAB BLOOD TEST (Slot 2)", style_td_mono)],
        [Paragraph("#3", style_td_mono), Paragraph("P-103", style_td_mono), Paragraph("79.3%", style_td_mono), Paragraph("183 mg/dL", style_td), Paragraph("23.3", style_td), Paragraph("32", style_td), Paragraph("PRIORITY LAB BLOOD TEST (Slot 3)", style_td_mono)],
        [Paragraph("#4", style_td_mono), Paragraph("P-109", style_td_mono), Paragraph("76.5%", style_td_mono), Paragraph("197 mg/dL", style_td), Paragraph("30.5", style_td), Paragraph("53", style_td), Paragraph("PRIORITY LAB BLOOD TEST (Slot 4)", style_td_mono)],
        [Paragraph("#5", style_td_mono), Paragraph("P-113", style_td_mono), Paragraph("75.9%", style_td_mono), Paragraph("139 mg/dL", style_td), Paragraph("27.1", style_td), Paragraph("57", style_td), Paragraph("PRIORITY LAB BLOOD TEST (Slot 5)", style_td_mono)],
        [Paragraph("#6", style_td_mono), Paragraph("P-101", style_td_mono), Paragraph("70.9%", style_td_mono), Paragraph("148 mg/dL", style_td), Paragraph("33.6", style_td), Paragraph("50", style_td), Paragraph("PRIORITY LAB BLOOD TEST (Slot 6 - Quota Limit)", style_td_mono)],
        [Paragraph("#7", style_td_mono), Paragraph("P-114", style_td_mono), Paragraph("66.3%", style_td), Paragraph("189 mg/dL", style_td), Paragraph("30.1", style_td), Paragraph("59", style_td), Paragraph("Routine Lifestyle Counseling & Home Referral", style_td)],
        [Paragraph("#19", style_td_mono), Paragraph("P-104", style_td_mono), Paragraph("4.0%", style_td), Paragraph("89 mg/dL", style_td), Paragraph("28.1", style_td), Paragraph("21", style_td), Paragraph("Routine Lifestyle Counseling & Home Referral", style_td)],
        [Paragraph("#20", style_td_mono), Paragraph("P-102", style_td_mono), Paragraph("4.0%", style_td), Paragraph("85 mg/dL", style_td), Paragraph("26.6", style_td), Paragraph("31", style_td), Paragraph("Routine Lifestyle Counseling & Home Referral", style_td)]
    ]
    t_q = Table(queue_data, colWidths=[35, 60, 55, 65, 45, 40, 200])
    t_q.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), COLOR_DARK),
        ('GRID', (0,0), (-1,-1), 0.5, COLOR_LINE),
        ('BACKGROUND', (0,1), (-1,6), COLOR_MINT),
        ('ROWBACKGROUNDS', (0,7), (-1,-1), [COLOR_WHITE, COLOR_MINT2]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_q)
    story.append(Spacer(1, 10))

    # ==============================================================================
    # 7. CLINICAL USER INTERFACE & VIVA DEFENSE
    # ==============================================================================
    story.append(Paragraph("7. Clinical Governance, Ethics & Viva Defense Questions", style_h1))
    story.append(Paragraph(
        "<b>Clinical Safety Limitations:</b> This tool is strictly an educational screening and triage prototype. It does not provide "
        "a diagnostic finding. All decisions must be confirmed with venous laboratory blood tests. Patients exhibiting acute emergency symptoms "
        "(severe ketoacidosis, extreme polyuria, dizziness) must bypass the model and receive immediate emergency physician care.",
        style_body
    ))
    story.append(Spacer(1, 6))

    viva_qa = [
        ("Q1: Why did KNN win over Decision Tree and Logistic Regression under the 30% constraint?",
         "While Decision Tree had higher raw accuracy at T=0.50 (79.2%), the business metric is Recall@30%. Distance-weighted KNN (k=9) effectively isolated non-linear clusters of high-risk patients, successfully detecting 31/54 actual diabetics (57.4% Recall) with 66.0% testing precision, capturing the largest volume of true positive disease."),
        ("Q2: Why not delete rows with zeros in Insulin and SkinThickness?",
         "Deleting rows with missing insulin or skin thickness discards over 50% of the patient cohort (376 out of 768 records). In medicine, deleting half the dataset causes massive sample size destruction and introduces severe demographic selection bias. Median imputation retains all 768 patient records."),
        ("Q3: How does the model prevent Data Leakage?",
         "We used a unified scikit-learn Pipeline. The imputer and StandardScaler were fitted exclusively on the 614 training instances and transformed onto the 154 holdout test instances. No statistical parameters from the test set ever contaminated the training distributions.")
    ]

    for q, a in viva_qa:
        story.append(Paragraph(f"<b>{q}</b>", style_body_bold))
        story.append(Paragraph(a, style_body))
        story.append(Spacer(1, 4))

    # Build PDF
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[SUCCESS] PDF Generated successfully at: {output_pdf_path}")

if __name__ == "__main__":
    out_path = os.path.join("outputs", "Diabetes_Risk_Prediction_Major_Project_Complete_Documentation.pdf")
    build_pdf_documentation(out_path)

