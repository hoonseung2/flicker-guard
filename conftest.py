# smoke_test.py (untracked, repo root) is a manual verification script --
# see docs/superpowers/specs/2026-08-03-mitigator-design.md section 7
# ("구현 완료 후 수동 검증 (자동 테스트 아님)": manual verification after
# implementation, not an automated test). It has no test_* functions, so it
# contributes nothing to the suite, but pytest's default rootdir collection
# still imports it (its name matches *_test.py) and executes its top-level
# code -- including a real training epoch over data/synthetic/ -- purely as
# an import side effect. That both wastes minutes on every `pytest` run and
# now breaks collection outright: train_one_epoch/evaluate no longer move
# the model to `device` themselves (see the --resume device-mismatch fix in
# training/train_mitigator.py), and this script never calls model.to(device)
# itself, so it raises a device-mismatch RuntimeError during collection.
# Excluded here rather than edited, since it isn't part of this fix wave.
collect_ignore = ["smoke_test.py"]
