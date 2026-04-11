# Mac Dev Setup for AI/Python Startup Developer

## 1. Foundation — Package Manager

**Install Homebrew first. Everything else flows from it.**

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

---

## 2. Terminal Stack

### Terminal Emulator
**Ghostty** (modern, fast, native Mac) or **Warp** (AI-assisted, great for beginners).
- Ghostty: `brew install --cask ghostty`
- Warp: `brew install --cask warp`

### Shell: ZSH + Oh My Zsh
ZSH is already default on Mac. Add Oh My Zsh for plugins and themes:

```bash
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```

### Essential ZSH Plugins
Add these to `plugins=(...)` in `~/.zshrc`:

```bash
plugins=(git z zsh-autosuggestions zsh-syntax-highlighting)
```

Install the two external ones:
```bash
brew install zsh-autosuggestions zsh-syntax-highlighting
```

### Prompt: Starship
Clean, fast prompt that shows git branch, Python version, virtualenv — everything a startup dev needs at a glance.

```bash
brew install starship
echo 'eval "$(starship init zsh)"' >> ~/.zshrc
```

---

## 3. Git Workflow Tools

### Core
```bash
brew install git git-lfs
git config --global user.name "Your Name"
git config --global user.email "you@email.com"
git config --global init.defaultBranch main
```

### TUI Git Client: Lazygit
The single biggest productivity multiplier for git. Staging hunks, rebasing, resolving conflicts — all visual, all in the terminal.
```bash
brew install lazygit
```
Open with `lg` (alias it in `.zshrc`):
```bash
alias lg='lazygit'
```

### GitHub CLI
```bash
brew install gh
gh auth login
```
Use it to create PRs, review issues without leaving terminal:
```bash
gh pr create
gh issue list
```

### Diff Tool: Delta
Replaces the default `git diff` with syntax-highlighted, side-by-side diffs.
```bash
brew install git-delta
```
Add to `~/.gitconfig`:
```ini
[core]
    pager = delta
[delta]
    navigate = true
    side-by-side = true
```

### Commit Convention: Commitizen (optional but startup-impressive)
Enforces `feat:`, `fix:`, `chore:` style commits. Many startups require this.
```bash
pip install commitizen
```

---

## 4. Python Developer Stack

### Python Version Manager: pyenv
Never use system Python. `pyenv` lets you switch Python versions per project.
```bash
brew install pyenv
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
pyenv install 3.12.3
pyenv global 3.12.3
```

### Virtual Environments: uv (replaces pip + venv + pip-tools)
`uv` is the modern standard — it's 10–100x faster than pip and is what startups are adopting now.
```bash
brew install uv
```

Daily workflow:
```bash
uv init my-project        # new project
uv add langchain openai   # replaces pip install
uv run python main.py     # runs in isolated env
uv sync                   # sync from lockfile (like npm install)
```

### Linter + Formatter: Ruff
Replaces `black`, `flake8`, `isort` — all in one, extremely fast.
```bash
uv add --dev ruff
```
Add `ruff.toml` or configure in `pyproject.toml`:
```toml
[tool.ruff]
line-length = 88
select = ["E", "F", "I"]
```

### Type Checker: mypy or pyright
Startups building production AI systems care about types.
```bash
uv add --dev mypy
```

### Testing: pytest
```bash
uv add --dev pytest pytest-cov
```

### Environment Variables: python-dotenv
```bash
uv add python-dotenv
```
Never hardcode API keys. Always use `.env` + add `.env` to `.gitignore`.

---

## 5. Editor: VS Code

```bash
brew install --cask visual-studio-code
```

### Essential Extensions for AI/Python dev

| Extension | Why |
|---|---|
| Python (Microsoft) | Core Python support |
| Pylance | Type checking in-editor |
| Ruff | Lint/format on save |
| Jupyter | Notebook support |
| GitLens | Blame, history, PR links inline |
| GitHub Copilot | AI autocomplete (free for students) |
| Even Better TOML | For `pyproject.toml` editing |
| Thunder Client | API testing (lightweight Postman) |

Apply for **GitHub Student Developer Pack** — gives you Copilot Pro free.

---

## 6. Project Structure (Startup Standard)

Every Python project you show in interviews should look like this:

```
my-project/
├── src/
│   └── my_project/
│       ├── __init__.py
│       └── main.py
├── tests/
│   └── test_main.py
├── .env                  # never commit
├── .env.example          # commit this
├── .gitignore
├── pyproject.toml        # uv manages this
├── README.md
└── uv.lock
```

---

## 7. .gitignore Essentials

Always include for Python/AI projects:
```
.env
__pycache__/
*.pyc
.venv/
.mypy_cache/
.ruff_cache/
*.ipynb_checkpoints/
```

Use `brew install gibo` and `gibo dump Python macOS JetBrains >> .gitignore` to auto-generate.

---

## 8. One-time Setup Checklist

```
[ ] Homebrew installed
[ ] Ghostty or Warp installed
[ ] Oh My Zsh + starship prompt
[ ] pyenv → Python 3.12
[ ] uv installed
[ ] git configured + gh auth
[ ] lazygit installed
[ ] VS Code + extensions
[ ] GitHub Student Pack applied (free Copilot)
[ ] SSH key added to GitHub: ssh-keygen -t ed25519
```

---

## Priority Order

If overwhelmed, do it in this order:

1. `brew` → `pyenv` → `uv` (Python foundation — day 1)
2. `git` config → `gh` → `lazygit` (Git — day 1)
3. VS Code + Ruff + Pylance (Editor — day 1)
4. Starship + Oh My Zsh plugins (Terminal polish — day 2)
5. `mypy`, `pytest`, commit conventions (Production habits — before first job)

The `uv` + `ruff` + `pyenv` combo is what experienced engineers at AI startups use in 2025/2026. Showing up already using these signals you're production-aware, not just a notebook researcher.
