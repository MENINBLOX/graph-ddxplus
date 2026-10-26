"""Runs BED's own code on validation rows and writes one JSON line per patient: questions, IL, final posterior and
BED's own metrics. It made tests/fixtures/bed_original_validate_head.json.gz (rows 0-999) and the whole-validation
check of the B-BED port (design 보완 2, 개발 전 확정 E).

Not part of the paper1 package and never run by it. It needs a checkout of mila-iqia/Casande-RL at 5279013, whose
`Baselines/BED/dataset/ddxplus/` JSON files (this directory's stats file among them) are copied next to a
`release_validate_patients.zip` that holds the validation CSV, in a separate environment:

    uv venv --python 3.12 bed_venv && VIRTUAL_ENV=bed_venv uv pip install numpy==1.26.4 pandas==2.2.3 tqdm==4.66.5
    bed_venv/bin/python -I run_original.py <Casande-RL>/Baselines/BED <input dir relative to it> out.jsonl 0 132448 22

The arguments are those `Baselines/BED/run_ddxplus.sh` passes (solver bed_batch, KL utility, search depth 1,
30 questions, threshold 0.01, pathology restriction on).

usage: python -I run_original.py BED_DIR INPUT_DIR_REL_TO_BED OUT.jsonl START STOP WORKERS
"""

import json
import multiprocessing
import sys
import time
import traceback
from argparse import Namespace

BED_DIR, INPUT_DIR, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
START, STOP, WORKERS = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])

_solver = None


def _init(bed_dir: str, input_dir: str) -> None:
    global _solver
    import numpy as np

    # numpy removed the np.float alias in 1.24; BED's utility reads np.finfo(np.float), the builtin float.
    np.float = float
    sys.path.insert(0, bed_dir)
    from bayesian_experimental_design_batch import BEDBatch

    args = Namespace(
        input_dir=input_dir, output_dir="output", dataset_name="DDXPlus_validate", n_diseases=200,
        solver="bed_batch", test_size=10000, max_episode_len=30, param_search=False, no_patho_restriction=False,
        utility_func="KL", threshold=0.01, search_depth=1, cheater_method="inference",
        single_diag_action=True, mask_actions=False, aux_reward=False,
    )
    _solver = BEDBatch(args)


def _run(i: int) -> dict:
    s = _solver
    df = s.qmr.test_data
    t = time.perf_counter()
    try:
        m = s.batch_run(df.iloc[i], s.max_episode_len)
    except Exception as e:
        return {"row": i, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-800:]}
    il = int(m["IL"])
    actions = [s.qmr.all_findings[a] for a in m["action_list"] if a != -1]
    final = m["diff_list"][:, il - 1]
    return {
        "row": i,
        "il": il,
        "q": actions[1:],
        "initial": actions[0],
        "posterior": {d: float(p) for d, p in zip(s.qmr.all_diseases, final) if p > 0},
        "gtpa1": bool(m["GTPA@1"]),
        "gtpa": bool(m["GTPA"]),
        "per": float(m["PER"]),
        "ddf1": float(m["DDF1"]),
        "dsf1": float(m["DSF1"]),
        "seconds": round(time.perf_counter() - t, 3),
    }


def main() -> None:
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(WORKERS, initializer=_init, initargs=(BED_DIR, INPUT_DIR)) as pool, open(OUT, "w") as f:
        for rec in pool.imap(_run, range(START, STOP), chunksize=4):
            f.write(json.dumps(rec) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
