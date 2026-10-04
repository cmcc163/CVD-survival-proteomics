from __future__ import annotations

import math
import os
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt


BASE_DIR = Path(__file__).resolve().parent
SHAP_ROOT = Path(os.environ.get("CVD_SHAP_ROOT", "artifacts/shap/final"))
MAPLE_ROOT = Path(os.environ.get("CVD_MAPLE_ROOT", "artifacts/genetics/maple"))
COLOC_FILE = Path(os.environ.get("CVD_COLOC_FILE", "artifacts/genetics/coloc_susie_results.xlsx"))
COLOC_SUSIE_ROOT = Path(os.environ.get("CVD_COLOC_ROOT", "artifacts/genetics/coloc_susie"))

OUTPUT_XLSX = BASE_DIR / "Supplementary_Table_MAPLE_and_SuSiE_colocalization.xlsx"
OUTPUT_DOCX = BASE_DIR / "Supplementary_Table_MAPLE_and_SuSiE_colocalization.docx"
OUTPUT_DOCX_FALLBACK = BASE_DIR / "Supplementary_Table_MAPLE_and_SuSiE_colocalization_updated.docx"

CLINICAL_VARS = {
    "age",
    "BMI",
    "hypertension_treatment",
    "Cholesterol_treatment",
    "sex",
    "average_SBP",
    "eGFR",
    "hdl_cholesterol",
    "ever_smoked",
    "Diabetes_baseline",
    "non_hdl_cholesterol",
}

MODEL_CONFIG = {
    "Total CVD": {
        "selection_file": SHAP_ROOT / "Total_CVD" / "Combined" / "Selection_Results_Total_CVD_Combined.xlsx",
        "maple_endpoints": ["CAD", "Stroke", "HF"],
    },
    "ASCVD": {
        "selection_file": SHAP_ROOT / "ASCVD" / "Combined" / "Selection_Results_ASCVD_Combined.xlsx",
        "maple_endpoints": ["CAD", "Stroke"],
    },
    "HF": {
        "selection_file": SHAP_ROOT / "HF" / "Combined" / "Selection_Results_HF_Combined.xlsx",
        "maple_endpoints": ["HF"],
    },
}


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell_border(cell, **kwargs) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_borders = tc_pr.first_child_found_in("w:tcBorders")
    if tc_borders is None:
        tc_borders = OxmlElement("w:tcBorders")
        tc_pr.append(tc_borders)
    for edge, attrs in kwargs.items():
        element = tc_borders.find(qn(f"w:{edge}"))
        if element is None:
            element = OxmlElement(f"w:{edge}")
            tc_borders.append(element)
        for key, value in attrs.items():
            element.set(qn(f"w:{key}"), str(value))


def clear_borders(cell) -> None:
    nil = {"val": "nil"}
    set_cell_border(cell, top=nil, bottom=nil, left=nil, right=nil, insideH=nil, insideV=nil)


def style_paragraph(paragraph, font_size=6.5, bold=False) -> None:
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(0)
    paragraph.paragraph_format.line_spacing = 1.0
    for run in paragraph.runs:
        run.font.name = "Arial"
        run._element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
        run.font.size = Pt(font_size)
        run.bold = bold


def set_cell_text(cell, text, font_size=6.5, bold=False) -> None:
    cell.text = "/" if pd.isna(text) else str(text)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    for paragraph in cell.paragraphs:
        style_paragraph(paragraph, font_size=font_size, bold=bold)
        paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT


def set_landscape(document: Document) -> None:
    section = document.sections[0]
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width, section.page_height = section.page_height, section.page_width
    section.top_margin = Inches(0.45)
    section.bottom_margin = Inches(0.45)
    section.left_margin = Inches(0.35)
    section.right_margin = Inches(0.35)


def add_heading(document: Document, prefix: str, title: str) -> None:
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(6)
    run = paragraph.add_run(prefix)
    run.bold = True
    run.font.name = "Arial"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
    run.font.size = Pt(10)
    run = paragraph.add_run(title)
    run.bold = True
    run.font.name = "Arial"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
    run.font.size = Pt(10)


def add_note(document: Document, text: str) -> None:
    note = document.add_paragraph()
    note.paragraph_format.first_line_indent = Inches(0.18)
    note.paragraph_format.space_before = Pt(6)
    note.add_run(text)
    style_paragraph(note, font_size=7.0)


def add_page_break(document: Document) -> None:
    document.add_page_break()


def add_table(document: Document, data: pd.DataFrame, widths: list[float]) -> None:
    table = document.add_table(rows=1, cols=len(data.columns))
    table.autofit = False

    for i, column in enumerate(data.columns):
        cell = table.rows[0].cells[i]
        cell.width = Inches(widths[i])
        set_cell_shading(cell, "BFBFBF")
        clear_borders(cell)
        set_cell_border(
            cell,
            top={"val": "single", "sz": "12", "color": "000000"},
            bottom={"val": "single", "sz": "8", "color": "000000"},
        )
        set_cell_text(cell, column, font_size=6.4, bold=True)

    prev_group = None
    group_columns = [c for c in ["Predictive outcome", "MAPLE endpoint", "Outcome"] if c in data.columns]
    for _, row in data.iterrows():
        cells = table.add_row().cells
        current_group = tuple(row[c] for c in group_columns) if group_columns else None
        for i, column in enumerate(data.columns):
            cells[i].width = Inches(widths[i])
            clear_borders(cells[i])
            set_cell_text(cells[i], row[column], font_size=6.1)
            if prev_group is not None and current_group != prev_group:
                set_cell_border(cells[i], top={"val": "single", "sz": "4", "color": "A6A6A6"})
        prev_group = current_group

    for cell in table.rows[-1].cells:
        set_cell_border(cell, bottom={"val": "single", "sz": "12", "color": "000000"})


def read_model_proteins(selection_file: Path, predictive_outcome: str) -> pd.DataFrame:
    data = pd.read_excel(selection_file, sheet_name="1_Ensemble_SHAP_Rank")
    data = data.loc[~data["Feature"].astype(str).isin(CLINICAL_VARS)].copy()
    data = data.sort_values(["Rank"])
    data["Protein rank"] = range(1, len(data) + 1)
    return data.rename(
        columns={
            "Feature": "Protein",
            "Rank": "Combined model rank",
            "Ensemble_Mean_SHAP_pct": "Mean SHAP (%)",
        }
    )[["Protein", "Combined model rank", "Protein rank", "Mean SHAP (%)", "Final_80pct_LRT", "Final_Kneedle_LRT"]].assign(
        **{"Predictive outcome": predictive_outcome}
    )


def bh_adjust(pvalues: pd.Series) -> pd.Series:
    p = pvalues.astype(float)
    out = pd.Series([math.nan] * len(p), index=p.index)
    valid = p.notna()
    if not valid.any():
        return out
    pv = p[valid].sort_values()
    m = len(pv)
    ranks = pd.Series(range(1, m + 1), index=pv.index)
    adjusted = (pv * m / ranks).iloc[::-1].cummin().iloc[::-1].clip(upper=1.0)
    out.loc[adjusted.index] = adjusted
    return out


def read_maple_results() -> pd.DataFrame:
    frames = []
    for endpoint in ["CAD", "Stroke", "HF"]:
        path = MAPLE_ROOT / endpoint / "MAPLE_all_results.xlsx"
        data = pd.read_excel(path, sheet_name="MAPLE_all_results")
        data = data.rename(
            columns={
                "protein": "Protein",
                "causal_effect": "Effect",
                "cause_se": "SE",
                "causal_pvalue": "Original P value",
                "FDR": "Original FDR",
            }
        )
        data["MAPLE endpoint"] = endpoint
        data["Z"] = data["Effect"] / data["SE"]
        data["P value"] = data["Z"].abs().map(lambda z: math.erfc(z / math.sqrt(2)) if pd.notna(z) else math.nan)
        data["FDR"] = bh_adjust(data["P value"])
        data["CI low"] = data["Effect"] - 1.96 * data["SE"]
        data["CI high"] = data["Effect"] + 1.96 * data["SE"]
        frames.append(
            data[
                [
                    "MAPLE endpoint",
                    "Protein",
                    "nsnp",
                    "Effect",
                    "SE",
                    "CI low",
                    "CI high",
                    "P value",
                    "FDR",
                    "Direction",
                    "Original P value",
                    "Original FDR",
                ]
            ]
        )
    return pd.concat(frames, ignore_index=True)


def fmt4(x) -> str:
    if pd.isna(x):
        return "/"
    return f"{float(x):.4f}"


def fmte(x) -> str:
    if pd.isna(x):
        return "/"
    return f"{float(x):.3e}"


def fmt_int(x) -> str:
    if pd.isna(x):
        return "/"
    return str(int(x))


def fmt_pp(x) -> str:
    if pd.isna(x):
        return "/"
    value = float(x)
    if value == 0:
        return "<1.0e-300"
    if abs(value) < 1e-4:
        return f"{value:.2e}"
    return f"{value:.4f}"


def format_maple_table(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw.loc[raw["Effect"].notna()].copy()
    out = out.drop_duplicates(subset=["Protein", "MAPLE endpoint"])
    out = out.sort_values(["MAPLE endpoint", "Protein"])
    out["Exposure"] = out["Protein"]
    out["Outcome"] = out["MAPLE endpoint"]
    out["nsnp"] = out["nsnp"].map(fmt_int)
    out["Beta"] = out["Effect"].map(fmt4)
    out["SE"] = out["SE"].map(fmt4)
    out["P-value"] = out["P value"].map(fmte)
    out["FDR-corrected P-value"] = out["FDR"].map(fmte)
    return out[
        [
            "Outcome",
            "Exposure",
            "nsnp",
            "Beta",
            "SE",
            "P-value",
            "FDR-corrected P-value",
        ]
    ]


def build_maple_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    maple = read_maple_results()
    rows = []
    for predictive_outcome, config in MODEL_CONFIG.items():
        proteins = read_model_proteins(config["selection_file"], predictive_outcome)
        for endpoint in config["maple_endpoints"]:
            merged = proteins.merge(
                maple.loc[maple["MAPLE endpoint"] == endpoint],
                how="left",
                on="Protein",
            )
            merged["MAPLE endpoint"] = endpoint
            merged["Availability"] = merged["Effect"].notna().map(
                {True: "Available", False: "No eligible SNPs or no MAPLE result"}
            )
            rows.append(merged)
    raw = pd.concat(rows, ignore_index=True)
    raw = raw.sort_values(["Predictive outcome", "MAPLE endpoint", "Protein rank"])
    return raw, format_maple_table(raw)


def read_coloc_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    status_frames = []
    best_rows = []
    xl = pd.ExcelFile(COLOC_FILE)
    for endpoint in ["CAD", "Stroke", "HF"]:
        sheet = next(
            (
                s
                for s in xl.sheet_names
                if s.startswith(endpoint) and "p12" not in s and "覆盖" not in s
            ),
            None,
        )
        if sheet is not None:
            data = pd.read_excel(COLOC_FILE, sheet_name=sheet)
        else:
            summary_path = COLOC_SUSIE_ROOT / endpoint / f"{endpoint}_coloc_susie_summary.csv"
            if not summary_path.exists():
                continue
            data = pd.read_csv(summary_path)
            data["evidence_class"] = data["max_PP.H4"].apply(
                lambda x: "SuSiE colocalization (PP.H4 >= 0.80)" if pd.notna(x) and x >= 0.8 else "No primary SuSiE colocalization"
            )
            data["final_use"] = data["max_PP.H4"].apply(
                lambda x: "Supplementary result; not part of CAD/Stroke/HF main-text colocalization set"
                if pd.notna(x)
                else "No comparable signal-level SuSiE pair"
            )
        data = data.rename(
            columns={
                "outcome": "Outcome",
                "protein": "Protein",
                "status": "Status",
                "evidence_class": "Evidence class",
                "final_use": "Final use",
                "max_PP.H4": "Max PP.H4",
                "max_PP.H3": "Max PP.H3",
                "best_hit1": "SNP from exposure",
                "best_hit2": "SNP from outcome",
                "top_snp_for_best_H4": "Top SNP",
            }
        )
        status_frames.append(data)

        for _, row in data.iterrows():
            protein = str(row["Protein"])
            signal_file = COLOC_SUSIE_ROOT / endpoint / f"{protein}_{endpoint}_coloc_susie_signal_summary.csv"
            if not signal_file.exists() or signal_file.stat().st_size == 0:
                continue
            signals = pd.read_csv(signal_file)
            if signals.empty or "PP.H4.abf" not in signals.columns:
                continue
            best = signals.loc[signals["PP.H4.abf"].astype(float).idxmax()].copy()
            best_rows.append(
                {
                    "Outcome": endpoint,
                    "Exposure": protein,
                    "SNP from exposure": best.get("hit1"),
                    "SNP from outcome": best.get("hit2"),
                    "nsnps": best.get("nsnps"),
                    "PP.H0": best.get("PP.H0.abf"),
                    "PP.H1": best.get("PP.H1.abf"),
                    "PP.H2": best.get("PP.H2.abf"),
                    "PP.H3": best.get("PP.H3.abf"),
                    "PP.H4": best.get("PP.H4.abf"),
                    "Exposure signal index": best.get("idx1"),
                    "Outcome signal index": best.get("idx2"),
                    "Status": row.get("Status"),
                    "Evidence class": row.get("Evidence class"),
                    "Final use": row.get("Final use"),
                    "Interpretation": row.get("interpretation"),
                }
            )

    raw_status = pd.concat(status_frames, ignore_index=True)
    raw_status = raw_status[
        [
            "Outcome",
            "Protein",
            "Status",
            "Evidence class",
            "Final use",
            "Max PP.H4",
            "Max PP.H3",
            "SNP from outcome",
            "SNP from exposure",
            "Top SNP",
            "nsnps",
            "interpretation",
        ]
    ].copy()
    raw_status["SuSiE primary positive"] = raw_status["Max PP.H4"].astype(float).ge(0.8).fillna(False)
    raw_status = raw_status.sort_values(
        ["Outcome", "SuSiE primary positive", "Max PP.H4", "Protein"],
        ascending=[True, False, False, True],
    )

    raw_best = pd.DataFrame(best_rows)
    raw_best["SuSiE primary positive"] = raw_best["PP.H4"].astype(float).ge(0.8).fillna(False)
    raw_best = raw_best.sort_values(
        ["Outcome", "SuSiE primary positive", "PP.H4", "Exposure"],
        ascending=[True, False, False, True],
    )

    formatted = raw_best.copy()
    formatted["nsnps"] = formatted["nsnps"].map(fmt_int)
    for column in ["PP.H0", "PP.H1", "PP.H2", "PP.H3", "PP.H4"]:
        formatted[column] = formatted[column].map(fmt_pp)
    for column in ["SNP from outcome", "SNP from exposure", "Interpretation"]:
        formatted[column] = formatted[column].fillna("/")
    formatted = formatted[
        [
            "Outcome",
            "Exposure",
            "SNP from outcome",
            "SNP from exposure",
            "nsnps",
            "PP.H0",
            "PP.H1",
            "PP.H2",
            "PP.H3",
            "PP.H4",
        ]
    ]
    return raw_status, raw_best, formatted


def write_outputs(
    maple_raw: pd.DataFrame,
    maple_fmt: pd.DataFrame,
    coloc_status: pd.DataFrame,
    coloc_best: pd.DataFrame,
    coloc_fmt: pd.DataFrame,
) -> None:
    summary = (
        maple_raw.groupby(["Predictive outcome", "MAPLE endpoint"])
        .agg(
            n_proteins=("Protein", "count"),
            n_available=("Effect", lambda x: x.notna().sum()),
            n_missing=("Effect", lambda x: x.isna().sum()),
            n_fdr_0_05=("FDR", lambda x: (x <= 0.05).sum()),
        )
        .reset_index()
    )
    coloc_summary = (
        coloc_status.groupby("Outcome")
        .agg(
            n_pairs=("Protein", "count"),
            n_primary_positive=("SuSiE primary positive", "sum"),
            primary_positive_proteins=(
                "Protein",
                lambda s: "; ".join(s[coloc_status.loc[s.index, "SuSiE primary positive"]]) or "/",
            ),
        )
        .reset_index()
    )

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        maple_fmt.to_excel(writer, sheet_name="Supplementary Table 9", index=False)
        maple_raw.to_excel(writer, sheet_name="MAPLE raw recalculated", index=False)
        maple_raw.loc[maple_raw["Effect"].isna()].to_excel(writer, sheet_name="MAPLE omitted pairs", index=False)
        summary.to_excel(writer, sheet_name="MAPLE summary", index=False)
        coloc_fmt.to_excel(writer, sheet_name="Supplementary Table 10", index=False)
        coloc_best.to_excel(writer, sheet_name="SuSiE best H4 signal pair", index=False)
        coloc_status.to_excel(writer, sheet_name="SuSiE primary status", index=False)
        coloc_summary.to_excel(writer, sheet_name="SuSiE summary", index=False)

    document = Document()
    set_landscape(document)
    add_heading(
        document,
        "Supplementary Table 9. ",
        "MAPLE genetic association results for proteins selected by the Total CVD, ASCVD, and HF prediction models",
    )
    add_table(
        document,
        maple_fmt,
        widths=[0.82, 1.18, 0.62, 0.78, 0.72, 1.18, 1.42],
    )
    add_note(
        document,
        "Protein-endpoint pairs are ordered by outcome and exposure. Duplicate protein-endpoint pairs selected by more than one prediction model are listed once. "
        "P values were recalculated from Beta/SE using a two-sided normal test; FDR-corrected P values were calculated within each MAPLE endpoint using the Benjamini-Hochberg method. "
        "Beta and SE are shown to four decimal places; P values are shown in scientific notation. "
        "CAD indicates coronary artery disease; CVD, cardiovascular disease; FDR, false discovery rate; HF, heart failure; MAPLE, Mendelian randomization analysis using probabilistic latent effects; SE, standard error."
    )
    add_page_break(document)
    add_heading(
        document,
        "Supplementary Table 10. ",
        "SuSiE colocalization results for protein-outcome pairs",
    )
    add_table(
        document,
        coloc_fmt,
        widths=[0.60, 0.86, 1.05, 1.05, 0.50, 0.74, 0.74, 0.74, 0.68, 0.68],
    )
    add_note(
        document,
        "SuSiE colocalization was performed with p1 = 1e-4, p2 = 1e-4, and p12 = 1e-5. "
        "For each protein-outcome pair, the signal pair with the largest PP.H4 is shown. PP.H0-PP.H4 denote posterior probabilities for no association, association with the protein only, association with the outcome only, two distinct causal variants, and one shared causal variant, respectively. "
        "Only pairs with comparable SuSiE signal-level configurations are listed; no such configuration was identified for HF under the primary analysis setting. "
        "Very small underflowed posterior probabilities are shown as <1.0e-300."
    )
    try:
        document.save(OUTPUT_DOCX)
    except PermissionError:
        document.save(OUTPUT_DOCX_FALLBACK)


def main() -> None:
    maple_raw, maple_fmt = build_maple_table()
    coloc_status, coloc_best, coloc_fmt = read_coloc_table()
    write_outputs(maple_raw, maple_fmt, coloc_status, coloc_best, coloc_fmt)
    print(f"Wrote {OUTPUT_XLSX}")
    if OUTPUT_DOCX.exists():
        print(f"Wrote {OUTPUT_DOCX}")
    if OUTPUT_DOCX_FALLBACK.exists():
        print(f"Wrote {OUTPUT_DOCX_FALLBACK}")


if __name__ == "__main__":
    main()
