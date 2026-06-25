from __future__ import annotations

import csv
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "doc" / "related_papers_for_comparison"


PAPERS = [
    {
        "method": "KnowSAM",
        "title": "Learnable Prompting SAM-induced Knowledge Distillation for Semi-supervised Medical Image Segmentation",
        "github": "https://github.com/taozh2017/KnowSAM",
        "paper_page": "https://arxiv.org/abs/2412.13742",
        "pdf_url": "https://arxiv.org/pdf/2412.13742.pdf",
        "filename": "01_KnowSAM_Learnable_Prompting_SAM_induced_KD.pdf",
        "note": "Current project baseline.",
    },
    {
        "method": "PH-Net",
        "title": "PH-Net: Semi-Supervised Breast Lesion Segmentation via Patch-wise Hardness",
        "github": "https://github.com/jjjsyyy/PH-Net",
        "paper_page": "https://openaccess.thecvf.com/content/CVPR2024/html/Jiang_PH-Net_Semi-Supervised_Breast_Lesion_Segmentation_via_Patch-wise_Hardness_CVPR_2024_paper.html",
        "pdf_url": "https://openaccess.thecvf.com/content/CVPR2024/papers/Jiang_PH-Net_Semi-Supervised_Breast_Lesion_Segmentation_via_Patch-wise_Hardness_CVPR_2024_paper.pdf",
        "filename": "02_PH_Net_Patch_wise_Hardness_CVPR2024.pdf",
        "note": "Hard-patch mining and ultrasound boundary difficulty baseline.",
    },
    {
        "method": "CPC-SAM",
        "title": "Cross Prompting Consistency with Segment Anything Model for Semi-supervised Medical Image Segmentation",
        "github": "https://github.com/JuzhengMiao/CPC-SAM",
        "paper_page": "https://papers.miccai.org/miccai-2024/170-Paper0321.html",
        "pdf_url": "https://papers.miccai.org/miccai-2024/paper/0321_paper.pdf",
        "filename": "03_CPC_SAM_Cross_Prompting_Consistency_MICCAI2024.pdf",
        "note": "Cross-prompt consistency reference.",
    },
    {
        "method": "SemiSAM+",
        "title": "SemiSAM+: Rethinking Semi-Supervised Medical Image Segmentation in the Era of Foundation Models",
        "github": "https://github.com/YichiZhang98/SemiSAM",
        "paper_page": "https://arxiv.org/abs/2502.20749",
        "pdf_url": "https://arxiv.org/pdf/2502.20749.pdf",
        "filename": "04_SemiSAMPlus_Foundation_Model_SSL.pdf",
        "note": "Foundation-model-driven semi-supervised learning reference.",
    },
    {
        "method": "SAM-MedUS",
        "title": "SAM-MedUS: a foundational model for universal ultrasound image segmentation",
        "github": "https://github.com/tf2bb/SAM-MedUS",
        "paper_page": "https://doi.org/10.1117/1.JMI.12.2.027001",
        "pdf_url": "https://www.spiedigitallibrary.org/journals/journal-of-medical-imaging/volume-12/issue-2/027001/SAM-MedUS--a-foundational-model-for-universal-ultrasound-image/10.1117/1.JMI.12.2.027001.full.pdf",
        "filename": "05_SAM_MedUS_Universal_Ultrasound_Segmentation.pdf",
        "manual_only": True,
        "note": "Ultrasound foundation model; NCBI OA API reports the PMCID is not Open Access, and SPIE blocks automated PDF retrieval. Keep DOI/GitHub for manual access.",
    },
    {
        "method": "E-BayesSAM",
        "title": "E-BayesSAM: Efficient Bayesian Adaptation of SAM with Self-Optimizing KAN-Based Interpretation for Uncertainty-Aware Ultrasonic Segmentation",
        "github": "https://github.com/mp31192/E-BayesSAM",
        "paper_page": "https://arxiv.org/abs/2508.17408",
        "pdf_url": "https://arxiv.org/pdf/2508.17408.pdf",
        "filename": "06_E_BayesSAM_Uncertainty_Aware_Ultrasonic_Segmentation.pdf",
        "note": "Uncertainty-aware SAM adaptation reference.",
    },
]


def download(url: str, path: Path, timeout: int = 60) -> tuple[bool, str]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0 Safari/537.36"
        ),
        "Accept": "application/pdf,text/html;q=0.9,*/*;q=0.8",
    }
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read()
            content_type = response.headers.get("content-type", "")
        if len(data) < 1024:
            return False, f"response too small ({len(data)} bytes)"
        if b"%PDF" not in data[:2048] and "pdf" not in content_type.lower():
            preview = data[:120].decode("utf-8", errors="replace").replace("\n", " ")
            return False, f"not a PDF response; content-type={content_type}; preview={preview}"
        path.write_bytes(data)
        return True, f"downloaded {len(data)} bytes"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}: {exc.reason}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def write_readme(rows: list[dict]):
    lines = [
        "# Related Papers for Comparison",
        "",
        "This folder stores papers associated with the GitHub repositories used in the KnowSAM A3 ultrasound semi-supervised segmentation report.",
        "",
        "## Download Status",
        "",
        "| Method | PDF | GitHub | Paper page | Status |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        pdf = row["local_pdf"] if row["downloaded"] else ""
        lines.append(
            f"| {row['method']} | {pdf} | {row['github']} | {row['paper_page']} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `papers_manifest.csv` and `papers_manifest.json` contain the same mapping in machine-readable form.",
            "- If a publisher blocks automated PDF retrieval, the row keeps the official paper page and GitHub link for manual access.",
            "- These papers are for local academic reading and experiment planning only; follow each publisher and repository license for redistribution.",
        ]
    )
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for paper in PAPERS:
        pdf_path = OUT_DIR / paper["filename"]
        if paper.get("manual_only"):
            ok = False
            status = "manual access required; DOI/GitHub recorded"
            if pdf_path.exists():
                pdf_path.unlink()
        elif pdf_path.exists() and pdf_path.stat().st_size > 1024:
            ok = True
            status = f"exists ({pdf_path.stat().st_size} bytes)"
        else:
            ok, status = download(paper["pdf_url"], pdf_path)
            if not ok and pdf_path.exists():
                pdf_path.unlink()
            time.sleep(1)

        row = {
            **paper,
            "downloaded": ok,
            "status": status,
            "local_pdf": paper["filename"] if ok else "",
        }
        row.pop("manual_only", None)
        rows.append(row)
        print(f"[{'OK' if ok else 'MISS'}] {paper['method']}: {status}")

    csv_path = OUT_DIR / "papers_manifest.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "title",
                "github",
                "paper_page",
                "pdf_url",
                "filename",
                "downloaded",
                "status",
                "local_pdf",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    (OUT_DIR / "papers_manifest.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_readme(rows)
    print(f"Manifest written to: {csv_path}")


if __name__ == "__main__":
    main()
