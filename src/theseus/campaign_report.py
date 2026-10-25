"""Review draft of the pooled campaign; all estimates come from recorded analysis."""

from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .common import load_json
from .report import REFERENCES


def render_campaign_report(directory):
    directory = Path(directory)
    out = directory / "analysis"
    record = load_json(directory / "campaign.json")
    spec, base = record["identity"]["spec"], record["identity"]["base"]
    summary = load_json(out / "summary.json")
    effects = load_json(out / "primary_effects.json")
    mock = base["inference"]["backend"] == "mock"
    label = "MOCK / SCRIPTED VALIDATION ONLY" if mock else "RESEARCH DRAFT - REVIEW BEFORE SUBMISSION"
    font_dir = Path(matplotlib.get_data_path()) / "fonts/ttf"
    for name, filename in [("TheseusSans", "DejaVuSans.ttf"), ("TheseusSans-Bold", "DejaVuSans-Bold.ttf")]:
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, str(font_dir / filename)))
    styles = getSampleStyleSheet()
    for style in styles.byName.values():
        if isinstance(style, ParagraphStyle):
            style.fontName = "TheseusSans-Bold" if "Bold" in style.fontName else "TheseusSans"
    styles.add(ParagraphStyle(name="Small", fontName="TheseusSans", fontSize=8, leading=10, spaceAfter=6))
    styles.add(ParagraphStyle(name="Body", fontName="TheseusSans", fontSize=9, leading=12, spaceAfter=7))
    styles["Title"].fontSize, styles["Title"].leading = 23, 27
    styles["Heading1"].fontSize = 14
    width = A4[0] - 80
    story = []

    def para(text, style="Body"):
        story.append(Paragraph(escape(text), styles[style]))

    def figure(name, height):
        if (out / name).exists():
            image = Image(str(out / name))
            scale = min(width / image.imageWidth, height / image.imageHeight)
            image.drawWidth, image.drawHeight = image.imageWidth * scale, image.imageHeight * scale
            story.append(image)
            story.append(Spacer(1, 6))

    abstract = (
        "Does replacing every pricing agent remove a high-profit convention when successors inherit its observations? "
        "We study complete model turnover in the supplied Apart/Poseidon pricing environment, with corrected currency-scaled equilibrium benchmarks. "
        f"The design includes {len(spec['models'])} fixed checkpoints, {2 * len(spec['pairs'])} directed replacement pairs, and separate neutral and explicitly instructed founder regimes. "
        "Matched branches vary inherited histories, replacement timing, and truthful private or public replacement notices. "
        "First-action disclosure outcomes are measured before new feedback diverges. "
        "Six predeclared comparisons measure history-reset effects, public versus private first prices, and public versus administrative-placebo persistence. "
        "A one-round best-response probe is compared with copying, holding-price, and myopic-response countermodels. "
        f"The recorded campaign contains {len(summary['complete_balanced_waves'])} complete balanced seed waves; incomplete waves remain visible and are excluded from pooled primary inference. "
        "The tables report paired effects with multiplicity-adjusted intervals and context-level uncertainty. "
        "Retail observations provide descriptive contexts and scales, not identified demand or cartel labels. "
        "Disclosure sensitivity cannot establish common knowledge, and recovery after a deviation cannot uniquely identify strategic punishment. "
        + (
            "All numerical results in this document are scripted software-validation outputs, not model evidence."
            if mock
            else "The recorded results require trace review and substantive interpretation before submission; no causal mechanism or real-market prevalence is assumed."
        )
    )
    (out / "abstract_draft.txt").write_text(abstract + "\n")
    para("The Cartel After Its Members", "Title")
    para("Cross-model inheritance and replacement disclosure", "Heading2")
    para(label, "Heading2")
    para(
        base.get("report", {}).get("author", "Author to verify")
        + " | "
        + base.get("report", {}).get("affiliation", "Affiliation to verify"),
        "Small",
    )
    para("Abstract", "Heading1")
    para(abstract)
    para("1. Predeclared pooled comparisons", "Heading1")
    figure("primary_effects.png", 250)
    para(
        "Intervals use Bonferroni tails across all six comparisons; p-values use Holm correction. Profit gain and first price have different normalizations. Read each endpoint separately. A zero-centered uncertain effect is inconclusive, not proof of equivalence.",
        "Small",
    )
    story.append(PageBreak())
    para("2. Effect estimates and transfer structure", "Heading1")
    labels = {
        "history_reset": "History - joint reset (gain)",
        "public_vs_private_first_price": "Public - private (first price)",
        "public_vs_placebo_persistence": "Public - placebo (gain)",
    }
    rows = [["Regime / contrast", "Effect", "Adjusted interval", "Holm p", "Contexts"]]
    for r in effects:
        rows.append(
            [
                r["regime"] + ": " + labels[r["hypothesis"]],
                f"{r['effect']:.4f}" if r["available"] else "Unavailable",
                f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]" if r["available"] else "-",
                f"{r['holm_p_value']:.4g}" if r["available"] else "-",
                str(r.get("clusters", 0)),
            ]
        )
    wrapped = [[Paragraph(escape(str(x)), styles["Small"]) for x in row] for row in rows]
    table = Table(wrapped, colWidths=[width * f for f in [0.40, 0.12, 0.24, 0.12, 0.12]], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 9))
    figure("transfer_matrix.png", 335)
    para(
        "Descriptive matrix: fixed pair and turnover direction, averaged over recorded repetitions. Forward replaces the first checkpoint in a pair with the second; reverse swaps that direction. This matrix is exploratory. Failures and incomplete cells are in coverage.csv; PC outcomes are in all_test_controls.csv.",
        "Small",
    )
    controls_path = out / "all_test_controls.csv"
    if controls_path.exists() and controls_path.stat().st_size > 3:
        controls = pd.read_csv(controls_path)
        controls = controls[controls.included_in_primary]
        success = controls[controls.ok]
        para(
            f"Test positive controls: {len(success)}/{len(controls)} executed successfully in included waves. Execution success is not behavioral success.",
            "Small",
        )
        for model, group in success.groupby("model"):
            para(
                f"{model}: PC mean terminal gain {group.terminal_gain.mean():.4f} ({len(group)} controls). These explicitly instructed outcomes are not evidence of neutral formation.",
                "Small",
            )
    story.append(PageBreak())
    para("3. Design and reproducibility", "Heading1")
    p = base["protocol"]
    para(
        f"{len(p['arms'])} branches share each founder trajectory. Founder horizons are selected on development contexts from {p['founder_round_candidates']}; the fallback is {p['founder_rounds']}. Sequential replacements occur at rounds 0 and {p['turnover_gap']}; matched simultaneous replacements occur at {p['turnover_gap']}. Follow-up lasts {p['post_turnover_rounds']} rounds, with the final {p['terminal_window']} scored. Every test founder is retained regardless of formation."
    )
    para(
        "Notice branches have identical histories at simultaneous replacement. Silent, administrative-placebo, private replacement, and public replacement conditions use the same successor checkpoint. Three prewritten variants are assigned by the trial seed. Both private recipients receive the fact without a statement about the other's receipt; the public condition explicitly discloses shared delivery. Notices repeat throughout follow-up. Wording and salience are not fully matched."
    )
    para(
        "Both agents choose from the same completed-round information. Own profits and quantities plus both prices are visible. The full demand formula and both sellers' costs and qualities are supplied. There is no scratchpad or persistent hidden agent process. Native checkpoint chat templates feed a common price-only grammar. These are substantive deviations from other pricing-agent prompt setups."
    )
    para(
        "Two BF16 engines run at a time. Capacity uses pinned checkpoint metadata and measured development-prompt tokens; request concurrency is chosen from a finite benchmark. Sampling and scientific parameters are fixed, not globally optimized. The global deadline persists across resume. Next waves start only when measured full-wave time plus declared headroom fits; this resource-based stopping rule is recorded."
    )
    para(
        "Pairing includes model pair, direction, context, and seed. Differences are averaged within product context, then products are resampled within dataset. Contexts share a structural game; these are not twelve independent real economies. Checkpoints are fixed cases, not a random model-family sample. Sign-flip tests require a symmetric null. Missingness can bias complete-case estimates."
    )
    para(
        "The pinned starter payoff scales price by alpha, but its FOC omits the corresponding derivative. Normalized-price solvers restore the factor; residual, finite-difference, and scale-invariance tests check the correction. The supplied environment still computes payoffs."
    )
    para(
        "The original v1 run remains an exploratory baseline and is not pooled here. Exact revisions, source hashes, data provenance, calibration, response records, and failure gates accompany every component study. Component PDFs are diagnostics; the six comparisons above are the campaign's primary family."
    )
    story.append(PageBreak())
    para("Appendix: Limitations and Dual-Use / Ethical Considerations", "Heading1")
    para(
        "High profits are not proof of agreement or intent. History dependence can reflect copying or anchoring. Disclosure effects measure sensitivity to factual wording, not an identified common-knowledge mechanism. A deviation followed by recovery can be generated by nonstrategic oscillation. The countermodels are references, not validated classifiers."
    )
    para(
        "PC-instructed formation and neutral formation remain separate. A failed positive control limits interpretation of neutral nulls. No demand elasticity, real-world cartel prevalence, or deployed pricing outcome is estimated. Cross-family effects conflate architecture, training, and chat templates; the selected models and finite horizons limit generalization."
    )
    para(
        "This is an offline safety audit. Persistence after replacement would show why successor policies and inherited observations deserve evaluation. It does not establish a real-world method for cartel operation or evasion. Customer identities are not sent to models. No live prices are changed."
    )
    para("References and submission review", "Heading1")
    for citation, url in REFERENCES[:7]:
        para(citation, "Small")
        story.append(
            Paragraph(f'<link href="{escape(url)}" color="#1d4ed8">{escape(url)}</link>', styles["Small"])
        )
    para(
        "Further sources, model cards, data attribution, and the novelty comparison are in docs/REFERENCES.md and docs/CAMPAIGN_PROTOCOL.md. AI assistance was used for implementation and drafting. Review all traces and claims, add actual findings to the abstract, and transfer the result to the official sprint template before submission. This PDF does not certify template compliance.",
        "Small",
    )

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("TheseusSans", 8)
        canvas.drawString(40, 24, "CAMPAIGN DRAFT | " + ("MOCK ONLY" if mock else "RECORDED EXPERIMENTS"))
        canvas.drawRightString(A4[0] - 40, 24, str(doc.page))
        canvas.restoreState()

    path = out / "report_draft.pdf"
    SimpleDocTemplate(
        str(path),
        pagesize=A4,
        topMargin=34,
        bottomMargin=40,
        leftMargin=40,
        rightMargin=40,
        title="The Cartel After Its Members",
    ).build(story, onFirstPage=footer, onLaterPages=footer)
    return path
