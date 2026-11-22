import sys, json, time, numpy as np, torch
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[1] + "/experiments")
import run
flush = sys.argv[2] == "1"
cfg = dict(run.DEFAULTS, **json.loads(sys.argv[3]))
torch.set_num_threads(1)
print("flush supported:", torch.set_flush_denormal(flush))
t0 = time.time()
out = run.run_train(cfg)
pf = np.array(out["snapshots"][str(cfg["steps"])])
h = out["history"][-1]["train"]
# re-build final logits? report summary
print(json.dumps({"flush": flush, "absorbed": h["absorbed"], "mean": h["mean"], "min_p_min": float(np.min(out["min_p"])),
  "n_abs": int((pf < 0.01).sum()), "abs_at_hash": hash(tuple(out["absorbed_at"])), "decreases": out["decreases_total"],
  "sec": time.time() - t0}))
np.save(sys.argv[4], pf)
