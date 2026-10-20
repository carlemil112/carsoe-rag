"""Konverterer dokumentation (pdf/docx/odt/pptx) til renset markdown + index.jsonl."""
import argparse, json, re, shutil, subprocess, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

CONVERT_EXT = {".pdf", ".docx", ".odt", ".pptx"}
SKIP_NAME = ("sensitiv",)  # disse filer konverteres slet ikke
FLAGS = {                  # markeres i indexet, men konverteres
    "archived": ("arkiv",),
    "outdated": ("skal opdateres", "legacy", "/old/"),
}


def pandoc_path():
    p = shutil.which("pandoc")
    if p:
        return p
    try:
        import pypandoc
        return pypandoc.get_pandoc_path()
    except Exception:
        return None


def to_markdown(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        import pymupdf4llm
        return pymupdf4llm.to_markdown(str(path))
    pandoc = pandoc_path()
    if pandoc:
        r = subprocess.run(
            [pandoc, str(path), "-t", "gfm", "--wrap=none"],
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode != 0:
            raise RuntimeError(r.stderr[:300])
        return r.stdout
    if ext == ".docx":
        import mammoth
        with open(path, "rb") as f:
            return mammoth.convert_to_markdown(f).value
    raise RuntimeError("pandoc mangler (kræves til " + ext + ")")


def clean(md: str):
    md = md.replace("\u00a0", " ")
    md = re.sub(r"</?u>", "", md)
    md = re.sub(r"</?(span|div)[^>]*>", "", md)
    n = 0
    for pat in (r"!\[[^\]]*\]\([^)]*\)", r"<img[^>]*>", r"\*{0,2}==> picture .*? <==\*{0,2}"):
        md, k = re.subn(pat, "[billede]", md)
        n += k
    md = re.sub(r"^(#+)\s*\*\*(.*?)\*\*\s*$", r"\1 \2", md, flags=re.M)
    md = re.sub(r"[ \t]+$", "", md, flags=re.M)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip() + "\n", n


def process(job):
    src, rel, out_root = job
    src = Path(src)
    low = "/" + rel.lower()
    rec = {"source": rel, "ext": src.suffix.lower(),
           "flags": [k for k, pats in FLAGS.items() if any(p in low for p in pats)]}
    try:
        md, n_img = clean(to_markdown(src))
        out = Path(out_root) / (rel + ".md")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        rec.update(status="ok", out=str(out.relative_to(out_root)),
                   chars=len(md), images=n_img)
        if len(md) < 200:
            rec["flags"].append("tom_eller_scannet")
    except Exception as e:
        rec.update(status="error", error=str(e)[:300])
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="kildemappe (læses kun)")
    ap.add_argument("out", help="outputmappe (skal ligge udenfor kilden)")
    ap.add_argument("--limit", type=int, help="kun de første N filer (til test)")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()

    src = Path(a.src).expanduser().resolve()
    out = Path(a.out).expanduser().resolve()
    if out == src or src in out.parents:
        sys.exit("Output må ikke ligge i kildemappen.")
    out.mkdir(parents=True, exist_ok=True)

    jobs, skipped = [], []
    for p in sorted(src.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(src).as_posix()
        if p.name.startswith("~$"):
            reason = "låsefil"
        elif p.suffix.lower() not in CONVERT_EXT:
            reason = "filtype"
        elif any(s in p.name.lower() for s in SKIP_NAME):
            reason = "sensitiv"
        else:
            jobs.append((str(p), rel, str(out)))
            continue
        skipped.append({"source": rel, "ext": p.suffix.lower(),
                        "status": "skipped", "reason": reason})
    if a.limit:
        jobs = jobs[: a.limit]

    print(f"{len(jobs)} filer skal konverteres, {len(skipped)} springes over", flush=True)
    results = []
    if a.workers > 1:
        with ProcessPoolExecutor(a.workers) as ex:
            it = ex.map(process, jobs)
            for i, r in enumerate(it, 1):
                print(f"[{i}/{len(jobs)}] {r['status']}: {r['source']}", flush=True)
                results.append(r)
    else:
        for i, j in enumerate(jobs, 1):
            r = process(j)
            print(f"[{i}/{len(jobs)}] {r['status']}: {r['source']}", flush=True)
            results.append(r)

    with open(out / "index.jsonl", "w", encoding="utf-8") as f:
        for r in results + skipped:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ok = [r for r in results if r["status"] == "ok"]
    print("\nOK:", len(ok), "| fejl:", len(results) - len(ok))
    print("Arkiv-markerede:", sum("archived" in r["flags"] for r in results))
    print("Forældede-markerede:", sum("outdated" in r["flags"] for r in results))
    print("Tomme/scannede:", sum("tom_eller_scannet" in r["flags"] for r in results))
    print("Med billeder:", sum(r.get("images", 0) > 0 for r in ok))
    for r in results:
        if r["status"] == "error":
            print("FEJL:", r["source"], "-", r["error"])


if __name__ == "__main__":
    main()