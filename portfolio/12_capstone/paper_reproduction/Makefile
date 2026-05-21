# Paper-reproduction Makefile. Targets:
#   make reproduce  — run the full PPO clip-ε ablation
#   make quick      — CI smoke version (~30 s, 1 seed × 4k steps)
#   make clean      — remove generated artefacts

PY ?= python

.PHONY: reproduce quick clean

reproduce:
	$(PY) ppo_clip_ablation.py

quick:
	$(PY) ppo_clip_ablation.py --quick

clean:
	rm -f figure_clip_ablation.png findings.md.bak
	@# Restore the placeholder findings.md (idempotent).
	@if [ ! -s findings.md ] || ! grep -q "Auto-generated" findings.md; then \
	  echo "(placeholder findings.md left in place; re-run \`make reproduce\` to fill)"; \
	fi
