"""
Gerador Automatizado de Relatórios Técnicos em PDF — CardioIA Fase 4
Grupo: Data Health Solutions (FIAP)

Converte os 3 relatórios Markdown (`relatorios/*.md`) em documentos HTML estilizados
no padrão editorial acadêmico FIAP (com formatação matemática Unicode limpa e paginação
exata de até 2 páginas A4) e renderiza os PDFs correspondentes via Chromium Headless.
"""

import os
import re
import subprocess
import tempfile
import markdown

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RELATORIOS_DIR = os.path.join(ROOT_DIR, "relatorios")

CSS_STYLE = """
@page {
  size: A4;
  margin: 10mm 13mm 10mm 13mm;
}
body {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
  color: #0F172A;
  line-height: 1.36;
  font-size: 9.0pt;
  margin: 0;
  padding: 0;
}
h1 {
  font-size: 13pt;
  color: #ED145B;
  border-bottom: 2px solid #ED145B;
  padding-bottom: 3px;
  margin-top: 0;
  margin-bottom: 5px;
}
h2 {
  font-size: 10.4pt;
  color: #0F172A;
  background: #F8FAFC;
  border-left: 4px solid #ED145B;
  padding: 3px 8px;
  margin-top: 8px;
  margin-bottom: 5px;
}
h3 {
  font-size: 9.4pt;
  color: #1E293B;
  margin-top: 6px;
  margin-bottom: 3px;
}
p, li {
  text-align: justify;
  margin-top: 2px;
  margin-bottom: 3px;
}
ul, ol {
  margin-top: 2px;
  margin-bottom: 5px;
  padding-left: 18px;
}
hr {
  border: none;
  border-top: 1px solid #CBD5E1;
  margin: 6px 0;
}
table {
  width: 100%;
  border-collapse: collapse;
  margin: 6px 0;
  font-size: 8.1pt;
}
th {
  background-color: #0F172A;
  color: #FFFFFF;
  padding: 4px 5px;
  border: 1px solid #334155;
  text-align: center;
}
td {
  padding: 3.5px 5px;
  border: 1px solid #CBD5E1;
}
tr:nth-child(even) td {
  background-color: #F8FAFC;
}
img {
  max-width: 92%;
  max-height: 150px;
  display: block;
  margin: 5px auto;
  border-radius: 6px;
  border: 1px solid #CBD5E1;
}
code {
  background: #F1F5F9;
  padding: 1px 4px;
  border-radius: 3px;
  font-family: monospace;
  font-size: 8.2pt;
  color: #BE123C;
}
.math-block {
  background: #F8FAFC;
  border: 1px solid #E2E8F0;
  border-left: 3px solid #38BDF8;
  border-radius: 5px;
  padding: 5px 10px;
  margin: 5px 0;
  text-align: center;
  font-family: "Cambria Math", "STIX Two Math", Georgia, serif;
  font-size: 9.3pt;
  font-weight: 600;
  color: #0F172A;
}
"""


def _resolve_relative_images(md_text: str) -> str:
    def repl(match):
        alt = match.group(1)
        rel_path = match.group(2)
        abs_path = os.path.abspath(os.path.join(RELATORIOS_DIR, rel_path))
        return f"![{alt}](file://{abs_path})"

    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", repl, md_text)


def _clean_latex_to_html(text: str) -> str:
    """Converte expressões LaTeX em notação tipográfica HTML/Unicode limpa para PDF."""
    replacements = {
        r"$$\text{ICT} = \frac{D_C}{D_T} = \frac{\text{Diâmetro Transverso Máximo Cardíaco (MRD + MLD)}}{\text{Diâmetro Transverso Interno Máximo Torácico}}$$": (
            '<div class="math-block">ICT = D<sub>C</sub> / D<sub>T</sub> = '
            "(Diâmetro Transverso Máximo Cardíaco [MRD + MLD]) / (Diâmetro Transverso Interno Máximo Torácico)</div>"
        ),
        r"$$Z_c = \frac{X_c - \mu_c}{\sigma_c}, \qquad \mu = [0,485,\; 0,456,\; 0,406], \quad \sigma = [0,229,\; 0,224,\; 0,225]$$": (
            '<div class="math-block">Z<sub>c</sub> = (X<sub>c</sub> − μ<sub>c</sub>) / σ<sub>c</sub> &nbsp;&nbsp;|&nbsp;&nbsp; '
            "μ = [0,485; 0,456; 0,406], &nbsp; σ = [0,229; 0,224; 0,225]</div>"
        ),
        r"$$\text{DIR} = \frac{\min_{g} P(\hat{Y} = 1 \mid A = g)}{\max_{g} P(\hat{Y} = 1 \mid A = g)} \ge 0,80$$": (
            '<div class="math-block">DIR = min<sub>g</sub> P(Ŷ = 1 | A = g) / max<sub>g</sub> P(Ŷ = 1 | A = g) ≥ 0,80</div>'
        ),
        r"$$\text{EOD} = \bigl|P(\hat{Y} = 1 \mid Y = 1, A = \text{F}) - P(\hat{Y} = 1 \mid Y = 1, A = \text{M})\bigr| = |\text{Recall}_{\text{F}} - \text{Recall}_{\text{M}}|$$": (
            '<div class="math-block">EOD = |P(Ŷ = 1 | Y = 1, A = F) − P(Ŷ = 1 | Y = 1, A = M)| = |Recall<sub>F</sub> − Recall<sub>M</sub>|</div>'
        ),
        r"$\text{DPD} = |P(\hat{Y} = 1 \mid A = \text{F}) - P(\hat{Y} = 1 \mid A = \text{M})|$": (
            "<b>DPD = |P(Ŷ = 1 | A = F) − P(Ŷ = 1 | A = M)|</b>"
        ),
        r"$\text{ICT} \le 0,50$": "<b>ICT ≤ 0,50</b>",
        r"$\text{ICT} > 0,50$": "<b>ICT &gt; 0,50</b>",
        r"$\emptyset$": "∅",
        r"$1024 \times 1024 \times 1$": "1024 × 1024 × 1",
        r"$224 \times 224$": "224 × 224",
        r"$3 \times 224 \times 224$": "3 × 224 × 224",
        r"$20\times$": "20×",
        r"$D_C / D_T$": "D<sub>C</sub> / D<sub>T</sub>",
        r"$c \in \{R, G, B\}$": "<i>c</i> ∈ {R, G, B}",
        r"$X \in [0, 1]$": "<i>X</i> ∈ [0, 1]",
        r"$\pm 7^\circ$": "±7°",
        r"$\pm 4\%$": "±4%",
        r"$0,96\times$": "0,96×",
        r"$1,04\times$": "1,04×",
        r"$\pm 8\%$": "±8%",
        r"$\text{kVp}$": "kVp",
        r"$\text{mAs}$": "mAs",
        r"$256 \times 4 \times 4 = 4096$": "256 × 4 × 4 = 4096",
        r"$\rightarrow$": "→",
        r"$\text{weight\_decay} = 10^{-4}$": "weight_decay = 10<sup>−4</sup>",
        r"$\tau_F = 0,43$": "τ<sub>F</sub> = 0,43",
        r"$\tau_{\text{F}} = 0,43$": "τ<sub>F</sub> = 0,43",
        r"$\tau = 0,50$": "τ = 0,50",
        r"$0,49$": "0,49",
        r"$0,42 \le P(\hat{Y}=1) < 0,50$": "0,42 ≤ P(Ŷ = 1) &lt; 0,50",
        r"$+4\%$": "+4%",
        r"$+8\%$": "+8%",
        r"$\text{Recall}_{\text{F}}$": "Recall<sub>F</sub>",
        r"$\text{Recall}_{\text{M}}$": "Recall<sub>M</sub>",
        r"$\text{FNR}_{\text{F}}$": "FNR<sub>F</sub>",
        r"$\text{EOD} \le 0,05$": "EOD ≤ 0,05",
        r"$\text{DIR} \ge 0,80$": "DIR ≥ 0,80",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def build_pdfs():
    md_files = [
        "relatorio_parte1_preprocessamento.md",
        "relatorio_parte2_modelos_cnn.md",
        "relatorio_ir_alem1_governanca_etica.md",
    ]

    for fname in md_files:
        md_path = os.path.join(RELATORIOS_DIR, fname)
        pdf_path = os.path.join(RELATORIOS_DIR, fname.replace(".md", ".pdf"))

        with open(md_path, "r", encoding="utf-8") as f:
            raw_md = f.read()

        resolved_md = _resolve_relative_images(raw_md)
        cleaned_md = _clean_latex_to_html(resolved_md)
        body_html = markdown.markdown(cleaned_md, extensions=["tables", "fenced_code"])

        full_html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8" />
  <style>{CSS_STYLE}</style>
</head>
<body>
{body_html}
</body>
</html>"""

        with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as tmp:
            tmp.write(full_html)
            tmp_html_path = tmp.name

        try:
            cmd = [
                "chromium",
                "--headless",
                "--disable-gpu",
                "--no-sandbox",
                "--no-pdf-header-footer",
                f"--print-to-pdf={pdf_path}",
                f"file://{tmp_html_path}",
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            size_kb = os.path.getsize(pdf_path) / 1024.0
            print(f"✅ PDF gerado: {pdf_path} ({size_kb:.1f} KB)")
        finally:
            if os.path.exists(tmp_html_path):
                os.remove(tmp_html_path)


if __name__ == "__main__":
    build_pdfs()
