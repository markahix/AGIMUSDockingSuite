#!/usr/bin/env python3
"""
Unit test for state_reader.get_queue_view. Synthetic job files, no DB/network.
Run: python3 -I test_queue.py
"""
import json
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import state_reader as sr

FAIL = []


def check(c, m):
    print(f"  [{'PASS' if c else 'FAIL'}] {m}")
    if not c:
        FAIL.append(m)


def iso(dt):
    return dt.isoformat()


tmp = Path(tempfile.mkdtemp()) / "jobs"
for s in ("pending", "running", "completed", "failed"):
    (tmp / s).mkdir(parents=True)

now = datetime.now()


def write(state, job):
    (tmp / state / f"{job['job_id']}.json").write_text(json.dumps(job))


# a completed dock job that took 600 s — seeds the runtime estimate
write("completed", {"job_id": "dock_done", "type": "dock",
                    "submitted_at": iso(now - timedelta(hours=1)),
                    "started_at": iso(now - timedelta(minutes=20)),
                    "completed_at": iso(now - timedelta(minutes=10)),
                    "host": "Agimus", "result": "completed"})
# a running dock job started 5 min ago on Adenosine
write("running", {"job_id": "dock_live", "type": "dock",
                  "submitted_at": iso(now - timedelta(minutes=8)),
                  "started_at": iso(now - timedelta(minutes=5)),
                  "host": "Adenosine"})
# a pending dock job
write("pending", {"job_id": "dock_wait", "type": "dock",
                  "submitted_at": iso(now - timedelta(minutes=2))})
# a failed prep job
write("failed", {"job_id": "prep_bad", "type": "prep_receptor",
                 "submitted_at": iso(now - timedelta(minutes=30)),
                 "started_at": iso(now - timedelta(minutes=29)),
                 "completed_at": iso(now - timedelta(minutes=28)),
                 "host": "Agimus", "result": "failed"})

q = sr.get_queue_view(tmp)
rows = {r["job_id"]: r for r in q["rows"]}

print("counts:")
check(q["counts"] == {"running": 1, "pending": 1, "completed": 1, "failed": 1},
      f"counts per state ({q['counts']})")

print("running row:")
live = rows["dock_live"]
check(live["status"] == "running" and live["host"] == "Adenosine",
      "running job shows host")
check(290 <= live["elapsed_s"] <= 320, f"elapsed ~5min ({live['elapsed_s']}s)")
check(abs(live["expected_s"] - 600) < 1, f"expected ~600s from history ({live['expected_s']})")

print("pending row:")
wait = rows["dock_wait"]
check(wait["status"] == "pending" and wait["elapsed_s"] is None,
      "pending job has no elapsed")
check(abs(wait["expected_s"] - 600) < 1, f"pending expected ~600s ({wait['expected_s']})")
check(wait["host"] is None, "pending job has no host yet")

print("completed row:")
done = rows["dock_done"]
check(abs(done["elapsed_s"] - 600) < 1, f"completed elapsed = 600s ({done['elapsed_s']})")
check(done["expected_s"] is None, "finished job shows no expected")

print("ordering: running before pending before finished:")
order = [r["status"] for r in q["rows"]]
check(order.index("running") < order.index("pending") < order.index("completed"),
      f"status order ({order})")

print()
if FAIL:
    print(f"{len(FAIL)} FAILED")
    sys.exit(1)
print("all tests passed")
