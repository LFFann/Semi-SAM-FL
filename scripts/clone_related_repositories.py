from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "doc" / "related_repositories_for_comparison"


REPOSITORIES = [
    {
        "name": "KnowSAM",
        "url": "https://github.com/taozh2017/KnowSAM",
        "category": "current_baseline",
        "note": "Current project baseline; clone kept for upstream comparison.",
    },
    {
        "name": "PH-Net",
        "url": "https://github.com/jjjsyyy/PH-Net",
        "category": "paper_related",
        "note": "Patch-wise hardness semi-supervised ultrasound segmentation.",
    },
    {
        "name": "CPC-SAM",
        "url": "https://github.com/JuzhengMiao/CPC-SAM",
        "category": "paper_related",
        "note": "Cross prompting consistency with SAM.",
    },
    {
        "name": "SemiSAM",
        "url": "https://github.com/YichiZhang98/SemiSAM",
        "category": "paper_related",
        "note": "SemiSAM/SemiSAM+ implementation.",
    },
    {
        "name": "SAM-MedUS",
        "url": "https://github.com/tf2bb/SAM-MedUS",
        "category": "paper_related",
        "note": "Universal ultrasound segmentation foundation model repository.",
    },
    {
        "name": "E-BayesSAM",
        "url": "https://github.com/mp31192/E-BayesSAM",
        "category": "paper_related",
        "note": "Uncertainty-aware ultrasonic segmentation reference.",
    },
    {
        "name": "CPAC-SAM",
        "url": "https://github.com/JuzhengMiao/CPAC-SAM",
        "category": "additional_user_requested",
        "note": "User-requested comparison repository.",
    },
    {
        "name": "BCSI",
        "url": "https://github.com/taozh2017/BCSI",
        "category": "additional_user_requested",
        "note": "User-requested comparison repository.",
    },
    {
        "name": "SFR",
        "url": "https://github.com/ShumengLI/SFR",
        "category": "additional_user_requested",
        "note": "User-requested comparison repository.",
    },
    {
        "name": "CMC",
        "url": "https://github.com/med-air/CMC",
        "category": "additional_user_requested",
        "note": "User-requested comparison repository.",
    },
    {
        "name": "AD-MT",
        "url": "https://github.com/ZhenZHAO/AD-MT",
        "category": "additional_user_requested",
        "note": "User-requested comparison repository.",
    },
    {
        "name": "ABD",
        "url": "https://github.com/Star-chy/ABD",
        "category": "additional_user_requested",
        "note": "User-requested comparison repository.",
    },
    {
        "name": "DiffRect",
        "url": "https://github.com/CUHK-AIM-Group/DiffRect",
        "category": "additional_user_requested",
        "note": "User-requested comparison repository.",
    },
]


def run_git(args: list[str], cwd: Path | None = None, timeout: int = 180) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd) if cwd else None,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        return proc.returncode == 0, proc.stdout.strip()
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def safe_dir_name(name: str) -> str:
    return name.replace("/", "_").replace(":", "_")


def clone_or_record(repo: dict) -> dict:
    target = OUT_DIR / safe_dir_name(repo["name"])
    row = {
        **repo,
        "local_path": str(target.relative_to(ROOT)),
        "status": "",
        "head": "",
        "default_branch": "",
    }

    if target.exists():
        if (target / ".git").exists():
            ok, head = run_git(["rev-parse", "HEAD"], cwd=target, timeout=30)
            row["status"] = "exists"
            row["head"] = head if ok else ""
            ok_branch, branch = run_git(["branch", "--show-current"], cwd=target, timeout=30)
            row["default_branch"] = branch if ok_branch else ""
        else:
            row["status"] = "target exists but is not a git repository"
        return row

    ok, output = run_git(["clone", "--depth", "1", repo["url"], str(target)], timeout=300)
    if ok:
        row["status"] = "cloned"
        ok_head, head = run_git(["rev-parse", "HEAD"], cwd=target, timeout=30)
        row["head"] = head if ok_head else ""
        ok_branch, branch = run_git(["branch", "--show-current"], cwd=target, timeout=30)
        row["default_branch"] = branch if ok_branch else ""
    else:
        row["status"] = f"clone failed: {output}"
    return row


def write_readme(rows: list[dict]):
    lines = [
        "# Related Repositories for Comparison",
        "",
        "This folder stores shallow clones of GitHub repositories used for KnowSAM A3 ultrasound semi-supervised segmentation comparison work.",
        "",
        "## Clone Status",
        "",
        "| Repository | Category | Local path | Head | Status |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['category']} | {row['local_path']} | {row['head']} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Clones are shallow (`--depth 1`) to keep local storage manageable.",
            "- Re-run `python scripts/clone_related_repositories.py` to fill missing repositories or refresh the manifest.",
            "- Check each repository license before reusing code in experiments or publications.",
        ]
    )
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for repo in REPOSITORIES:
        row = clone_or_record(repo)
        rows.append(row)
        print(f"[{row['status']}] {repo['name']} -> {row['local_path']}")

    csv_path = OUT_DIR / "repositories_manifest.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "url",
                "category",
                "note",
                "local_path",
                "status",
                "head",
                "default_branch",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    (OUT_DIR / "repositories_manifest.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_readme(rows)
    print(f"Manifest written to: {csv_path}")


if __name__ == "__main__":
    main()
