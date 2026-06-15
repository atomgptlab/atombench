# Installation

Requires Python 3.9+.

```bash
pip install atombench            # metrics, figures, tables
pip install 'atombench[submit]'  # adds leaderboard submission (requests, jarvis-tools)
```

Submitting also needs `git` on your PATH and a GitHub token in `GITHUB_TOKEN`
(or `GH_TOKEN`) — see the
[tutorial](tutorial.md#3-submit-to-the-jarvis-leaderboard).

## From source

Install an editable copy to modify AtomBench or track the latest development version.

```bash
git clone git@github.com:atomgptlab/atombench.git
cd atombench
pip install -e '.[submit]'
```

## Verify

Confirm the install worked and the command-line tools are on your PATH.

```bash
atombench -h
atombench-submit -h
```
