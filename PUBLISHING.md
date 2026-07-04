# Publishing ContextWatch

## 1. Publish to GitHub

```bash
cd contextwatch
git init -b main            # already done if you followed the setup
git add -A
git commit -m "Initial release: context window profiler for LLM agents"

# Create the repo and push (requires `gh` CLI, or create it on github.com)
gh repo create contextwatch --public --source=. --push
```

Manual alternative: create an empty repo named `contextwatch` on github.com, then

```bash
git remote add origin https://github.com/sackri10/contextwatch.git
git push -u origin main
```

## 2. Publish to PyPI

### One-time setup

1. Create accounts at <https://pypi.org> and <https://test.pypi.org>.
2. Enable 2FA on both (required for new projects).
3. Create an **API token** on each: Account settings → API tokens →
   "Add API token" (scope: entire account for the first upload; you can
   re-scope to the project afterwards).
4. Install build tooling:

   ```bash
   pip install --upgrade build twine
   ```

### Every release

```bash
# 0. Bump the version in BOTH places (keep them in sync):
#    - pyproject.toml       -> version = "X.Y.Z"
#    - src/contextwatch/__init__.py -> __version__ = "X.Y.Z"

# 1. Clean old artifacts and run tests
rm -rf dist/ build/ src/*.egg-info
pytest

# 2. Build sdist + wheel
python -m build
# -> dist/contextwatch-X.Y.Z.tar.gz
# -> dist/contextwatch-X.Y.Z-py3-none-any.whl

# 3. Sanity-check the metadata
twine check dist/*

# 4. Dry run against TestPyPI first
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ --no-deps contextwatch
python -c "import contextwatch; print(contextwatch.__version__)"

# 5. Publish for real
twine upload dist/*
```

`twine upload` prompts for credentials: username is `__token__`, password is
the API token (starts with `pypi-`). To skip prompts, put the token in
`~/.pypirc`:

```ini
[pypi]
username = __token__
password = pypi-...

[testpypi]
username = __token__
password = pypi-...
```

### Verify

```bash
pip install contextwatch
contextwatch --help
```

## 3. Recommended: automate with GitHub Actions (trusted publishing)

PyPI supports **Trusted Publishers** — no tokens stored in CI:

1. On PyPI: project → Settings → Publishing → add GitHub publisher
   (owner `sackri10`, repo `contextwatch`, workflow `release.yml`).
2. Add `.github/workflows/release.yml`:

```yaml
name: release
on:
  release:
    types: [published]
jobs:
  pypi:
    runs-on: ubuntu-latest
    environment: release
    permissions:
      id-token: write        # trusted publishing
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.12"}
      - run: pip install build && python -m build
      - uses: pypa/gh-action-pypi-publish@release/v1
```

3. Cut a release: tag `vX.Y.Z` on GitHub → "Publish release" → CI uploads
   to PyPI automatically.

## Release checklist

- [ ] Version bumped in `pyproject.toml` and `__init__.py`
- [ ] `pytest` green
- [ ] `python examples/demo_agent.py` renders a report
- [ ] `twine check dist/*` passes
- [ ] TestPyPI install works
- [ ] Git tag `vX.Y.Z` pushed, GitHub release notes written
