"""Build a review draft from recorded results, never a fabricated submission."""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .common import load_json

REFERENCES = [
    (
        "Calvano, Calzolari, Denicolo and Pastorello (2020). Artificial Intelligence, Algorithmic Pricing, and Collusion. American Economic Review 110(10), 3267-3297.",
        "https://doi.org/10.1257/aer.20190623",
    ),
    (
        "Fish, Gonczarowski and Shorrer (2024; revised 2026). Algorithmic Collusion by Large Language Models.",
        "https://arxiv.org/abs/2404.00806",
    ),
    (
        "Eschenbaum, Mellgren and Zahn (2022). Robust Algorithmic Collusion.",
        "https://arxiv.org/abs/2201.00345",
    ),
    (
        "Keppo, Li, Tsoukalas and Yuan (2026). On the Fragility of AI Agent Collusion.",
        "https://arxiv.org/abs/2603.20281",
    ),
    (
        "Riemer et al. (2026). Position: Collusion Risks Among AI Reasoning Agents Justify Certification Requirements for Making Market Decisions.",
        "https://arxiv.org/abs/2608.18078",
    ),
    (
        "Shi, Zhang and Yang (2026). Emergent Collusion in Long-Horizon LLM Agent Interaction.",
        "https://arxiv.org/abs/2609.24967",
    ),
    (
        "Poseidon Research. Apart collusion_hackathon starter repository, commit e6fb1308dae9cf97ae42aec1c5f7889d9e2291e8.",
        "https://github.com/poseidon-research/collusion_hackathon",
    ),
    (
        "Chen. Online Retail II. UCI Machine Learning Repository, DOI 10.24432/C5CG6D.",
        "https://archive.ics.uci.edu/dataset/502/online+retail+ii",
    ),
    (
        "Kilts Center, University of Chicago Booth. Dominick's Finer Foods scanner data and codebook.",
        "https://www.chicagobooth.edu/research/kilts/research-data/dominicks",
    ),
]


def render_report(run):
    run = Path(run)
    meta = load_json(run / "preregistration.json")
    cfg, output = meta["config"], run / "analysis"
    p = cfg["protocol"]
    metrics = pd.read_csv(output / "episode_metrics.csv")
    effects = [x for x in load_json(output / "effects.json") if x["family"] == "primary"]
    controls = load_json(output / "positive_controls.json")
    diagnostic = (
        load_json(output / "deviation_summary.json")
        if (output / "deviation_summary.json").exists()
        else {"available": False}
    )
    mock = cfg["inference"]["backend"] == "mock"
    label = (
        "SCRIPTED / MOCK VALIDATION - NOT MODEL EVIDENCE"
        if mock
        else "RESEARCH DRAFT - REVIEW BEFORE SUBMISSION"
    )
    experiment = cfg.get("experiment", "neutral")
    title = "The Ship of Theseus Cartel"
    subtitle = (
        "PC-instructed founders, neutral successors"
        if experiment == "planted"
        else "Neutral founders, neutral successors"
    )
    styles = getSampleStyleSheet()
    # Embed the fonts shipped with the already-required Matplotlib dependency.
    # PDF viewers otherwise substitute base fonts with inconsistent glyph widths.
    font_dir = Path(matplotlib.get_data_path()) / "fonts/ttf"
    for name, filename in [("TheseusSans", "DejaVuSans.ttf"), ("TheseusSans-Bold", "DejaVuSans-Bold.ttf")]:
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, str(font_dir / filename)))
    for style in styles.byName.values():
        if isinstance(style, ParagraphStyle):
            style.fontName = "TheseusSans-Bold" if "Bold" in style.fontName else "TheseusSans"
    styles.add(
        ParagraphStyle(name="BodySmall", fontName="TheseusSans", fontSize=9.2, leading=12, spaceAfter=7)
    )
    styles.add(
        ParagraphStyle(
            name="CaptionSmall",
            fontName="TheseusSans",
            fontSize=8,
            leading=10,
            spaceAfter=7,
            textColor=colors.HexColor("#475569"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="Notice",
            fontName="TheseusSans-Bold",
            fontSize=9,
            leading=12,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#9f1239"),
            spaceAfter=8,
        )
    )
    styles["Title"].fontSize = 22
    styles["Title"].leading = 26
    styles["Heading1"].fontSize = 15
    styles["Heading1"].leading = 18
    styles["Heading2"].fontSize = 11
    width = A4[0] - 1.4 * inch
    story = []

    def para(text, style="BodySmall"):
        story.append(Paragraph(escape(str(text)), styles[style]))

    def heading(text):
        para(text, "Heading1")

    def figure(name, caption, height):
        file = output / name
        if file.exists():
            story.append(Image(str(file), width=width, height=height, kind="proportional"))
            para(caption, "CaptionSmall")

    def table(rows, widths):
        wrapped = [[Paragraph(escape(str(x)), styles["CaptionSmall"]) for x in row] for row in rows]
        t = Table(wrapped, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#64748b")),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.append(t)
        story.append(Spacer(1, 8))

    para(title, "Title")
    para(subtitle, "Heading2")
    para(
        cfg.get("report", {}).get("author", "Author to verify")
        + " | "
        + cfg.get("report", {}).get("affiliation", "Affiliation to verify"),
        "CaptionSmall",
    )
    para(label, "Notice")
    if cfg.get("campaign_context"):
        para(
            "Component block diagnostic. Use the campaign report for the six pooled primary comparisons.",
            "Notice",
        )
    count = len(metrics)
    abstract = (
        "Replacing an AI pricing agent may leave the observations that supported its behavior intact. "
        "We test whether coordinated high-profit pricing survives replacement of both founding agents, "
        "and whether sequential replacement differs from a simultaneous replacement made at the same final turnover time. "
        f"This {experiment} study branches identical founder histories into {len(p['arms'])} conditions, "
        "including inherited-history removal, synchronized resets, competitive counterfactual histories, and a one-round deviation probe. "
        f"The recorded dataset contains {count} completed post-founder branches. "
        "Primary outcomes are normalized profit gains relative to verified Nash and joint-monopoly benchmarks; "
        "paired differences are averaged within product contexts before uncertainty estimation. "
        "Explicitly instructed positive controls are retained and reported on the test contexts. "
        "The design reuses the supplied pricing environment while correcting a missing currency-scale derivative in its equilibrium solver. "
        "Retail records supply descriptive contexts and scales, while demand remains a structural assumption. "
        "The deviation probe measures response and recovery but does not uniquely identify strategic punishment. "
        "The estimates below therefore concern the specified simulator, prompts, checkpoints, and finite horizon, "
        "not real-world cartel prevalence or intent. "
        + (
            "All numerical values in this draft come from scripted software validation, not model experiments."
            if mock
            else "This automatically populated draft requires human review, contextual interpretation, and transfer to the official submission template."
        )
    )
    heading("Abstract")
    para(abstract)
    figure(
        "trajectories_primary.png",
        "Figure 1. Mean normalized profit gain. Both turnover schedules are scored over the same terminal window after complete replacement. Zero is the Nash benchmark; one is joint monopoly profit.",
        270,
    )
    story.append(PageBreak())
    heading("1. Results and positive controls")
    para(label, "Notice")
    para(
        "Primary comparisons are within the same founder trajectory. Intervals adjust bootstrap tails across the three primary contrasts. Seeds and directions are nested within product context; rounds are not independent samples."
    )
    rows = [["Contrast", "Effect", "Adjusted interval", "Holm p", "Contexts"]]
    for e in effects:
        rows.append(
            [
                e["left"].replace("sequential_", "seq_") + " vs " + e["right"].replace("sequential_", "seq_"),
                f"{e['effect']:.4f}" if e["available"] else "NA",
                f"[{e['ci_low']:.4f}, {e['ci_high']:.4f}]" if e["available"] else "NA",
                f"{e['holm_p_value']:.4g}" if e["available"] else "NA",
                str(e.get("clusters", 0)),
            ]
        )
    table(rows, [width * 0.39, width * 0.12, width * 0.24, width * 0.13, width * 0.12])
    successful = [r for r in controls if r["ok"]]
    para(
        f"Test positive controls completed: {len(successful)}/{len(controls)}. The PC instruction is explicitly collusion-seeking; its outputs are not evidence of spontaneous formation under neutral instructions."
    )
    if successful:
        control_rows = [["Control model", "Mean terminal gain", "Completed"]]
        for model, g in pd.DataFrame(successful).groupby("model"):
            control_rows.append([model, f"{g['terminal_gain'].mean():.4f}", len(g)])
        table(control_rows, [width * 0.53, width * 0.29, width * 0.18])
    para("Deviation response and recovery", "Heading2")
    if diagnostic["available"]:
        para(
            f"Matched probe trials: {diagnostic['paired_trials']}; actual same-round undercuts: {diagnostic['actual_undercut_trials']}. The other seller chooses before the forced action is visible and can respond only from the following round."
        )
        figure(
            "deviation_response.png",
            "Figure 2. Responder-price difference from its matched no-probe branch. Shading is pointwise product-cluster uncertainty, not a simultaneous band. Recovery or oscillation can arise from reactive anchoring.",
            250,
        )
    else:
        para("No complete matched deviation trials are available. No probe conclusion is supported.")
    story.append(PageBreak())
    heading("2. Design, interpretation, and benchmark correction")
    para(
        f"Founder instruction: {p['founder_prefix']}; successor instruction: {p['newcomer_prefix']}. Models: {cfg['models']['founder']['id']} and {cfg['models']['replacement']['id']}, with both turnover directions. Founder horizon: {p['founder_rounds']} rounds, selected on development contexts only. Sequential replacements occur at branch rounds 0 and {p['turnover_gap']}; the matched simultaneous replacement occurs at {p['turnover_gap']}. Follow-up: {p['post_turnover_rounds']} rounds; terminal scoring window: {p['terminal_window']}."
    )
    para(
        "The early incumbent/successor interaction intentionally differs between schedules. Matching final turnover time removes unequal post-turnover exposure from the primary contrast, but does not isolate an abstract timing effect independent of the histories generated during the gap."
    )
    para("Prompt and decoding are part of the treatment setting", "Heading2")
    para(
        "The observation explicitly names a two-seller simulation and supplies the demand formula, both costs and qualities, public prices, and own outcomes. There is no plans-and-insights scratchpad. Non-thinking decoding is constrained to a numeric price line. These choices can change behavior and must not be described as a faithful reproduction of the original Fish et al. prompt setup."
    )
    para("Verified currency-scale correction", "Heading2")
    para(
        "The pinned starter payoff uses p/alpha, but its price FOC omits the corresponding 1/alpha derivative. We solve normalized prices x=p/alpha, check residuals, and convert back into currency. The unchanged environment still computes actual demand and payoff."
    )
    para(
        "Nash: 1 - (x_i - c_i)(1 - s_i)/mu = 0. Joint monopoly: 1 - (x_i - c_i)/mu + sum_j((x_j - c_j)s_j)/mu = 0."
    )
    para(
        "Regression tests compare the original symmetric unit-scale solution, verify scale invariance, and check numerical profit derivatives. The correction is a reproducibility contribution, not a substitute for model evidence."
    )
    para("What the comparisons can establish", "Heading2")
    para(
        "The main estimates test persistence and history dependence under specified interventions. Positive persistence with a loss under synchronized resets is compatible with transmission through observations. Similar behavior can also reflect anchoring, shared priors, imitation, or pricing errors. The deviation diagnostic narrows the evidence but does not resolve these mechanisms uniquely."
    )
    para(
        "The inherited-histories and replacement-policy motivations build on Calvano et al. (2020), Fish et al. (2024), Eschenbaum et al. (2022), and Keppo et al. (2026). The specific proposed contribution is complete-turnover comparison with controlled history interventions, rather than a claim that collusion, heterogeneity, or memory effects are newly discovered."
    )
    story.append(PageBreak())
    heading("Appendix A. Limitations and Dual-Use / Ethical Considerations")
    para("Scope and external validity", "Heading2")
    para(
        "Real retail records calibrate descriptive context and scale in an assumed logit game; UCI contexts share the normalized economy and Dominick's cost proxies add limited variation, so twelve product clusters are not twelve independent empirical economies. No causal demand elasticities or real cartel labels are estimated."
    )
    para(
        "Shared priors can support similar pricing without inter-agent transmission. Cross-family comparisons also change training, architecture, and native chat templates. Price-only decoding, syntax constraints, finite horizons, initialization, and full-information observations limit generalization. PC-instructed founders must remain separate from neutral founders in every table and claim."
    )
    para("Statistics and missingness", "Heading2")
    para(
        "All valid founder histories enter the primary analysis, including low-profit histories. An established-only analysis is secondary. Product-cluster bootstrap intervals can be unstable with few contexts, and sign-flip tests assume a symmetric null. Complete-case comparisons can be biased if failures vary by arm. Invalid model outputs are cached, not repeatedly sampled until favorable; no missing action is imputed."
    )
    missing = load_json(output / "missing_episodes.json")
    para(
        f"Recorded missing test arms: {len(missing)}. The machine-readable quality gates, control failures, and per-arm traces must be inspected alongside aggregate effects. A passed software test is not evidence that the research has adequate power."
    )
    para("Anchoring and the deviation probe", "Heading2")
    para(
        "A one-round best-response action is calculated from the previously observed rival price. Near competitive prices it need not be a downward shock, so actual undercut counts are reported. A temporary price reduction and later recovery can occur under simple reactive rules; it is not proof of punishment, intent, or a repeated-game equilibrium."
    )
    para("Dual use and responsible interpretation", "Heading2")
    para(
        "The work is an offline audit. It changes no real prices, contacts no firms, and excludes customer identifiers from model inputs. A potential misuse is retaining a harmful pricing convention through agent replacement while presenting the replacement as remediation. The research motivation is to audit successors and inherited observations, not to improve live cartel operation. No real-world evasion capability is claimed."
    )
    para("Reproducibility and authorship", "Heading2")
    para(
        "Review preregistration.json, source and data hashes, hardware and concurrency measurements, exact model revisions, raw trajectories, and the validation record. AI assistance was used for implementation and drafting; the submitting author must verify all claims. Preexisting supplied code and literature are attributed. The official template could not be retrieved in this environment, so this PDF is a review draft, not a certified template-compliant submission."
    )
    story.append(PageBreak())
    heading("References")
    for n, (citation, url) in enumerate(REFERENCES, 1):
        para(f"[{n}] {citation}")
        story.append(
            Paragraph(
                f'<link href="{escape(url)}" color="#1d4ed8">{escape(url)}</link>', styles["CaptionSmall"]
            )
        )
    para("Official submission template and requirements", "Heading2")
    para(
        "Use the current template linked on the sprint page. It requires a PDF, a 150-250 word abstract, author details, and a Limitations and Dual-Use / Ethical Considerations appendix. The main report limit is eight pages, excluding references and appendices."
    )
    url = "https://apartresearch.com/sprints/ai-collusion-research-sprint-2026-10-23-to-2026-10-25"
    story.append(
        Paragraph(
            f'<link href="{url}" color="#1d4ed8">Sprint requirements and template link</link>',
            styles["CaptionSmall"],
        )
    )

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("TheseusSans", 8)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawString(
            0.7 * inch,
            0.4 * inch,
            "DRAFT | " + experiment + (" | MOCK / SCRIPTED ONLY" if mock else " | RECORDED MODEL RUN"),
        )
        canvas.drawRightString(A4[0] - 0.7 * inch, 0.4 * inch, str(doc.page))
        canvas.restoreState()

    document = SimpleDocTemplate(
        str(output / "report_draft.pdf"),
        pagesize=A4,
        rightMargin=0.7 * inch,
        leftMargin=0.7 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.65 * inch,
        title=title,
        author=cfg.get("report", {}).get("author", ""),
    )
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    (output / "abstract_draft.txt").write_text(abstract + "\n")
    return output / "report_draft.pdf"
