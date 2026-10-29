"""Build one-page English and Chinese summaries from verified result files."""

# Read saved results and use a small document layout library.
import json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from pypdf import PdfReader

# Resolve the repository and a font that can render both summary languages.
ROOT = Path(__file__).resolve().parents[1]
FONT_PATH = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
if FONT_PATH.exists():
    pdfmetrics.registerFont(TTFont("SummarySans", str(FONT_PATH)))
    FONT = "SummarySans"
else:
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    FONT = "STSong-Light"


# Build the same one-page structure for both languages.
def write_summary(language, results):
    metrics = {row["model"]: row for row in results["cv_metrics"] if row["scope"] == "all"}
    full, reduced, baseline = metrics["full"], metrics["reduced"], metrics["training_mean"]
    models = {(row["specification"], row["model"]): row for row in results["models"]}
    report = models[("report_with_year", "reduced")]
    pooled = models[("extension_pooled", "full")]
    chinese = language == "zh"

    # Keep every numerical claim linked to the actual computation.
    if chinese:
        title = "LLM API 价格：复现与独立扩展"
        subtitle = "快照：2026-03-31 | UW IS 451 课程小组项目的课后独立扩展"
        sections = [
            ("研究问题与数据", "哪些可观察的模型特征与付费 API 价格相关？固定 Kaggle 版本 1：453 行中，138 行价格缺失、16 行价格为零，留下 299 行正价格样本。价格单位为每百万 token 的美元混合价格；provider 列实际代表模型开发者。"),
            ("原结果复现", f"加回发布年份，沿用完整模型中 p&lt;0.05 的一次性筛选，自然得到 13 个解释项。精简模型 R²={report['r_squared']:.6f}，调整 R²={report['adjusted_r_squared']:.6f}，与报告的 0.575、0.555 相符；不含年份的 notebook 版 R²=0.566571。连续变量按 1 个标准差换算后的主要系数也对上报告。"),
            ("诊断与样本外检验", f"原模型含只有一行样本的开发者，杠杆值等于 1，HC3 无法可靠计算。独立扩展将训练样本少于 5 行的开发者合并，再做 HC3 与 5 折分组验证，共 {results['normalized_model_groups']} 个基础模型组。缺失值填补、变量筛选及价格回转修正均只使用训练折。"),
        ]
        headers = ["模型", "log MAE", "美元 MAE", "美元 RMSE", "样本外 log R²"]
        row_names = ["完整扩展", "一次筛选", "训练均值基准"]
        interpretation = f"完整扩展模型样本内 R²={pooled['r_squared']:.3f}，样本外降至 {full['oof_log_r2']:.3f}。筛选模型在美元误差上略低，但 log 误差更高；大额价格误差仍接近简单基准，不能把拟合价格当作可靠采购预算。"
        ranking = f"主排名覆盖 {results['ranking_eligible_rows']} 行：开发者在训练集出现、所有输入无需填补，且无已知年份错误。Gemma 3n E4B Instruct、Nova Lite 等低于预测基准；o3-pro 等高于基准。它们只是后续任务测试的候选信号，不代表质量优劣、公允价值或因果效果。"
        limits = "只有一个历史快照。参数量缺失 103 行、年份缺失 84 行；有 5 行已知年份错配，历史复现保留原值。p 值筛选属探索性分析；分组依赖名称规则，且仅做一次固定分折。价格规律不能证明某模型适合特定任务。Kaggle 无更新快照，未做两期比较。"
        labels = ["样本外误差（美元单位均为每百万 token）", "发现与业务含义", "价格偏离候选", "关键局限"]
        origin = "原课程贡献：453→299 数据筛选。新代码、诊断、分组验证及价格偏离分析为课后独立扩展。"
    else:
        title = "LLM API Pricing: Reproduction and Extension"
        subtitle = "March 31, 2026 snapshot | Individual extension after the UW IS 451 course"
        sections = [
            ("Question and data", "Which observable model features are associated with paid API prices? Kaggle version 1 contains 453 rows: 138 missing prices and 16 zero prices leave 299 paid models. The outcome is blended USD per million tokens. The source's provider field identifies the model creator, not necessarily the inference API provider."),
            ("Reproduction", f"Adding release year and applying the original one-pass p&lt;0.05 selection naturally retains 13 predictors. Reduced R²={report['r_squared']:.6f} and adjusted R²={report['adjusted_r_squared']:.6f} round to the report's 0.575 and 0.555. The no-year notebook version gives R²=0.566571. The report's main feature effects also match at its displayed precision."),
            ("Diagnostics and validation", f"Singleton creator indicators produce leverage of one, making historical HC3 inference unavailable. The extension pools creators with fewer than five training rows, then applies HC3 and five grouped folds over {results['normalized_model_groups']} base-model groups. Imputation, selection and log-to-dollar correction use training rows only."),
        ]
        headers = ["Procedure", "Log MAE", "USD MAE", "USD RMSE", "OOF log R²"]
        row_names = ["Full extension", "One-pass selected", "Training means"]
        interpretation = f"The full extension's in-sample R² of {pooled['r_squared']:.3f} falls to {full['oof_log_r2']:.3f} out of sample. Selection slightly lowers dollar errors but raises log errors. Large dollar errors remain close to the simple baseline, limiting the usefulness of fitted prices for budgeting."
        ranking = f"The main ranking covers {results['ranking_eligible_rows']} rows with seen creators, no imputed inputs and no known year error. Gemma 3n E4B Instruct and Nova Lite are below the prediction benchmark; o3-pro is above it. These are candidates for task-specific testing, not quality rankings, fair values or causal findings."
        limits = "One historical snapshot; 103 missing parameter counts and 84 missing release years. Five known year errors are preserved for reproduction. P-value selection is exploratory; family grouping is rule based and only one fixed fold assignment is used. No newer Kaggle snapshot was found, so no two-date comparison is reported."
        labels = ["Held-out errors (USD per million tokens)", "Finding and business implication", "Price-deviation candidates", "Key limitations"]
        origin = "Original course contribution: data screening from 453 to 299 rows. New code, diagnostics, validation and deviation analysis are the individual extension."

    # Set a compact page with restrained color and a visible source footer.
    body_style = ParagraphStyle("Body", fontName=FONT, fontSize=9.1, leading=12.6, textColor=colors.HexColor("#25364a"), wordWrap="CJK" if chinese else None)
    heading_style = ParagraphStyle("Heading", parent=body_style, fontSize=10.3, leading=13.5, textColor=colors.HexColor("#236e74"), spaceBefore=8, spaceAfter=3)
    title_style = ParagraphStyle("Title", parent=body_style, fontSize=17, leading=22, textColor=colors.HexColor("#203b59"), spaceAfter=4)
    small_style = ParagraphStyle("Small", parent=body_style, fontSize=7.6, leading=10, textColor=colors.HexColor("#637184"))
    story = [Paragraph(title, title_style), Paragraph(subtitle, small_style), Spacer(1, 7)]
    for heading, text in sections:
        story.extend([Paragraph(heading, heading_style), Paragraph(text, body_style)])

    # Display the same held-out procedures in a small results table.
    story.append(Paragraph(labels[0], heading_style))
    table_rows = [headers]
    for name, row in zip(row_names, [full, reduced, baseline]):
        table_rows.append([name, f"{row['log_mae']:.3f}", f"{row['price_mae']:.3f}", f"{row['price_rmse']:.3f}", f"{row['oof_log_r2']:.3f}"])
    table = Table(table_rows, colWidths=[150, 78, 78, 78, 100])
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), FONT), ("FONTSIZE", (0, 0), (-1, -1), 8.6),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaf0f5")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#25364a")),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, -1), (-1, -1), .5, colors.HexColor("#c7d2dd")),
    ]))
    story.append(table)
    for label, text in zip(labels[1:], [interpretation, ranking, limits]):
        story.extend([Paragraph(label, heading_style), Paragraph(text, body_style)])
    story.extend([Spacer(1, 9), Paragraph(origin, small_style), Spacer(1, 4), Paragraph('Source: <link href="https://www.kaggle.com/datasets/kanchana1990/llm-price-performance-tracker-march-2026" color="#315c88">Kaggle LLM Price-Performance Tracker, v1</link>. Source metadata checked 2026-10-28. See README for provenance and licensing limits.', small_style)])

    # Require a real one-page PDF instead of trusting the layout settings.
    destination = ROOT / "outputs" / f"summary_{language}.pdf"
    document = SimpleDocTemplate(str(destination), pagesize=letter, leftMargin=42, rightMargin=42, topMargin=30, bottomMargin=30, title=title, author="hql7-luo")
    document.build(story)
    if len(PdfReader(destination).pages) != 1:
        raise ValueError(f"{destination.name} must fit on exactly one page.")
    print(f"Created one page: {destination.name}")

    # Keep an editable text companion beside each PDF.
    lines = [f"# {title}", "", subtitle, ""]
    for heading, text in sections:
        lines.extend([f"## {heading}", "", text.replace("&lt;", "<"), ""])
    lines.extend([f"## {labels[0]}", "", " | ".join(headers), " | ".join(["---"] * 5)])
    lines.extend(" | ".join(row) for row in table_rows[1:])
    for label, text in zip(labels[1:], [interpretation, ranking, limits]):
        lines.extend(["", f"## {label}", "", text])
    lines.extend(["", origin, "", "Source: https://www.kaggle.com/datasets/kanchana1990/llm-price-performance-tracker-march-2026 (v1, metadata checked 2026-10-28).", ""])
    (ROOT / "outputs" / f"summary_{language}.md").write_text("\n".join(lines))


# Build both language versions from the same saved results.
if __name__ == "__main__":
    saved_results = json.loads((ROOT / "outputs/tables/results.json").read_text())
    for language in ["en", "zh"]:
        write_summary(language, saved_results)
