"""Wait for given PIDs to exit, then run queued commands sequentially (avoids pgrep self-matching)."""
import os
import subprocess
import sys
import time

pids = [int(p) for p in sys.argv[1].split(",")]
cmds = sys.argv[2:]
while any(os.path.exists(f"/proc/{p}") for p in pids):
    time.sleep(20)
for c in cmds:
    print("RUN", c, time.strftime("%H:%M:%S"), flush=True)
    rc = subprocess.call(c, shell=True)
    print("EXIT", rc, time.strftime("%H:%M:%S"), flush=True)
print("QUEUE_DONE", flush=True)
