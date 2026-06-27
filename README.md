# Multi-Objective Bayesian Optimization Practicum

Starter code for a 3-hour hands-on lab on **multi-objective decision making under
uncertainty**, viewed through the lens of Bayesian optimization. You will act as a
wet-lab antibody design team: each round you pick a batch of candidate sequences to
"test," and your goal is to grow the Pareto front (and its hypervolume) under a fixed
experimental budget.

The lab is designed to run **entirely on a CPU laptop** — no GPU required.

---

## 1. Setup (do this before the lab)

The project uses [**uv**](https://docs.astral.sh/uv/) to manage Python and all
dependencies. uv reads the checked-in `pyproject.toml` and `uv.lock` and builds an
identical, reproducible environment for everyone.

### Step 1 — Install uv

Pick the line for your operating system and run it in a terminal:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close and reopen your terminal afterward so `uv` is on your `PATH`. Check it:

```bash
uv --version
```

### Step 2 — Clone the repo

```bash
git clone https://github.com/jiwoncpark/multi-objective-reasoning-lab.git
cd multi-objective-reasoning-lab
```

### Step 3 — Build the environment

```bash
uv sync
```

That's it. `uv sync` will:

- download the correct Python interpreter (3.11) if you don't have it,
- create a virtual environment in `.venv/`,
- install the **CPU-only** build of PyTorch plus BoTorch, GPyTorch, Ax, Jupyter,
  and the plotting/data libraries — exactly as pinned in `uv.lock`.

You do **not** need to activate the virtual environment manually; prefix commands
with `uv run` (see below).

### Step 4 — Verify it works

```bash
uv run python -c "import torch, botorch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available()); print('botorch', botorch.__version__)"
```

Expected output (the exact versions may differ slightly):

```
torch 2.12.1+cpu | cuda: False
botorch 0.18.1
```

The `+cpu` suffix and `cuda: False` confirm you are on the CPU build — this is
correct and intended.

### Alternative: Google Colab (no local install)

If you can't install uv locally — for example on an Intel Mac (see Troubleshooting)
or a locked-down machine — you can run the lab in [Google Colab](https://colab.research.google.com)
from any browser. You only need a Google account.

1. Go to <https://colab.research.google.com> and choose **File → Open notebook → GitHub**.
2. Paste this repository's URL and open the notebook you want (start with
   `notebooks/00_pareto_hypervolume_warmup.ipynb`).
3. Add a **new code cell at the very top** and run this once per session to install
   the lab dependencies and make the helper code importable:

   ```python
   # Run this first in Colab (not needed for local uv setup)
   !pip install -q "botorch>=0.18.1" "gpytorch>=1.15.2" "ax-platform>=1.3.1"

   # Clone the repo so the data/ files and helper modules are available
   import os
   if not os.path.exists("multi-objective-reasoning-lab"):
       !git clone https://github.com/jiwoncpark/multi-objective-reasoning-lab.git
   %cd multi-objective-reasoning-lab
   ```

4. Run the notebook as normal.

Notes:

- Colab already includes PyTorch, NumPy, pandas, and matplotlib, so those are not
  reinstalled above.
- Package versions on Colab may differ slightly from the pinned `uv.lock`. The
  conceptual results are the same, but the exact "golden-path" numbers in
  Notebook 01 are only guaranteed with the local uv environment.
- Use a **CPU runtime** (the default). You do not need a GPU for this lab.

### If neither uv nor Colab works

Don't get stuck on setup. Pair up and **work alongside another student in your
group on their working machine** — this is a group lab, and sharing one
environment per pair is completely fine. Flag it to the instructor so we can help.

---

## 2. Running the notebooks

Launch JupyterLab through uv:

```bash
uv run jupyter lab
```

Then open the notebooks in order (they build on each other):

```
notebooks/
  00_pareto_hypervolume_warmup.ipynb
  01_seeded_noisy_batch_mobo_iteration.ipynb
  02_strategy_cards_practice.ipynb
  03_competition.ipynb
  04_optional_extensions.ipynb
```

To run a plain Python script instead:

```bash
uv run python path/to/script.py
```

---

## 3. Troubleshooting

| Symptom | Fix |
|---|---|
| `uv: command not found` | Reopen your terminal after installing uv, or add `~/.local/bin` (macOS/Linux) to your `PATH`. |
| `uv sync` is slow the first time | Normal — it's downloading PyTorch (~200 MB). It's cached afterward. |
| Kernel can't find a package in Jupyter | Make sure you launched with `uv run jupyter lab`, not a system Jupyter. |
| Want a clean rebuild | Delete the `.venv/` folder and run `uv sync` again. |

### Intel Mac users (older MacBooks)

The pinned CPU build of PyTorch 2.12 does **not** ship a wheel for Intel
(x86_64) Macs — only Apple Silicon (M1/M2/M3…). If `uv sync` fails on an Intel
Mac, use **Google Colab** instead (see
[Alternative: Google Colab](#alternative-google-colab-no-local-install) above), or
pair up with a teammate. Apple Silicon Macs, Windows, and Linux are all fully
supported locally.

---

## 4. For instructors / maintainers

The CPU-only PyTorch build is enforced in `pyproject.toml`:

```toml
[tool.uv.sources]
torch = [{ index = "pytorch-cpu" }]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

If you re-resolve dependencies (`uv lock`), keep this override so student
laptops stay GPU-free. To develop on a GPU machine instead, remove the override
and re-lock — but commit the CPU lock for distribution to students.
