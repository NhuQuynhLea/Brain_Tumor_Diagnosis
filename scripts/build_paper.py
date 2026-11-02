"""Build paper PDF using tectonic.
Supports both LaTeX2e+Proceedings+Template+ZIP and paper folders.
Run anytime main.tex is modified:
    python scripts/build_paper.py
"""
import os
import subprocess
import sys

def build():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tectonic_bin = os.path.join(repo_root, "tools", "tectonic.exe")

    # Find the target folder
    candidate_folders = [
        os.path.join(repo_root, "LaTeX2e+Proceedings+Template+ZIP"),
        os.path.join(repo_root, "paper"),
    ]
    target_dir = None
    for folder in candidate_folders:
        if os.path.exists(os.path.join(folder, "main.tex")):
            target_dir = folder
            break

    if not target_dir:
        print("Error: Could not locate main.tex in paper or template directory.", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(tectonic_bin):
        print(f"Error: tectonic executable not found at {tectonic_bin}", file=sys.stderr)
        sys.exit(1)

    tex_path = os.path.join(target_dir, "main.tex")
    out_pdf = os.path.join(target_dir, "main.pdf")

    # Run publication figures generator if venv python available or if figures need updating
    venv_py = os.path.join(repo_root, ".venv", "Scripts", "python.exe")
    py_exec = venv_py if os.path.exists(venv_py) else sys.executable
    try:
        subprocess.run([py_exec, os.path.join(repo_root, "scripts", "publication_figures.py")],
                       cwd=repo_root, check=False)
    except Exception:
        pass
    print(f"Compiling {tex_path} -> {out_pdf}...")
    cmd = [tectonic_bin, tex_path, "--outdir", target_dir]
    res = subprocess.run(cmd, cwd=repo_root)

    if res.returncode == 0 and os.path.exists(out_pdf):
        size_kb = os.path.getsize(out_pdf) / 1024
        print(f"Success! Generated {out_pdf} ({size_kb:.1f} KB)")
    else:
        print(f"Build failed with exit code {res.returncode}", file=sys.stderr)
        sys.exit(res.returncode if res.returncode != 0 else 1)

if __name__ == "__main__":
    build()
