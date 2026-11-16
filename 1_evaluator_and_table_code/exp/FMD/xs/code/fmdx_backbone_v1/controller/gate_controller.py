#!/usr/bin/env python3
"""FMDX gate-and-priority controller (stdlib only; runs on the cluster host, not in a container).

One process per cluster: --site clusterA (CLUSTER-A CPU job) and --site clusterB (CLUSTER-B CPU job). Every --period seconds:
  1. verdicts: CLUSTER-A reads exp/FMD/xs/verdicts/verdict_<fm>.json (the only place they are written) and pushes them, with
     SHA256SUMS_hub.txt, arms.tsv and the computed priority.json, to the VPS hub (stage/exp/FMD/xs/hub/).
     CLUSTER-B pulls the hub copy, verifies every sha256 against SHA256SUMS_hub.txt (any mismatch -> nothing applied, logged),
     and installs the verdicts under its mirror exp/FMD/xs/verdicts/ (the path its arm jobs read).
  2. gate: every arm whose FM verdict status is FAIL -> scancel all of this cluster's jobs named fx_<arm> (pending and
     running, incl. CLUSTER-B continuation segments), write runs/<arm>_<site>/CANCELLED_GATE_FAIL.json.
  3. races: owner of stage/exp/FMD/xs/race/<arm>/owner.txt (created atomically by the first copy to START) is read; this
     cluster's copies of an arm owned by the other lane are scancelled ('race lost').
  4. priority: the A4.3(b) rule (controls on top, FM arms by verdict priority_score desc) ranks the remaining arms; pending jobs get nice = NICE_BASE + NICE_STEP * rank via
     scontrol update; arms flagged hold_until_pass are held while their FM verdict is not PASS (scontrol hold / release).
Every action is one line in exp/FMD/xs/admin/controller_log.md (verdict, gate and race lines also in
exp/CAND/queue/cand_log.md on CLUSTER-A). CLUSTER-B lines go to its controller_log_clusterB.md, are pushed to the hub (hub_clusterB/), and
the CLUSTER-A controller appends unseen lines to both CLUSTER-A logs (dedup by line id).
Only jobs named fx_<arm> of arms listed in arms.tsv and owned by this user are ever touched.
"""
import argparse
import base64
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

VPS = "Administrator@<ip>"
HUB_WIN = "$VPS_ROOT/uhwr/stage/exp/FMD/xs"
NICE_BASE, NICE_STEP = 100, 100


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha(p):
    h = hashlib.sha256()
    with open(str(p), "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run(cmd, inp=None, timeout=120):
    try:
        r = subprocess.run(cmd, input=inp, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except Exception as e:
        return 99, b"", repr(e).encode()


class Ctl:
    def __init__(self, a):
        self.a = a
        self.site = a.site
        self.root = Path(a.root)
        self.xs = self.root / "exp/FMD/xs"
        self.adm = self.xs / "admin"
        self.adm.mkdir(parents=True, exist_ok=True)
        self.vdir = self.xs / "verdicts"
        self.log_path = self.adm / ("controller_log.md" if self.site == "clusterA" else "controller_log_clusterB.md")
        self.cand = self.root / "exp/CAND/queue/cand_log.md" if self.site == "clusterA" else None
        self.state_p = self.adm / ("controller_state_%s.json" % self.site)
        self.state = json.loads(self.state_p.read_text()) if self.state_p.exists() else {}
        for k, v in (("seen_clusterB_lines", []), ("verdict_sha", {}), ("cancelled", {})):
            self.state.setdefault(k, v)
        self.key = os.path.expanduser("~/.ssh/uhwr_staging_vps_ed25519")
        self.ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-i", self.key, VPS]
        self.user = os.environ.get("USER", "")

    # ------------------------------------------------------------------ logging
    def log(self, kind, msg, cand=False):
        lid = hashlib.sha256(("%s|%s|%s|%s" % (self.site, utc(), kind, msg)).encode()).hexdigest()[:10]
        line = "- %s [%s] [%s] %s (id %s)\n" % (utc(), self.site, kind, msg, lid)
        with open(str(self.log_path), "a") as f:
            f.write(line)
        if cand and self.cand is not None:
            with open(str(self.cand), "a") as f:
                f.write("- %s FMDX controller [%s] %s\n" % (utc(), self.site, msg))
        print(line, end="", flush=True)

    def save_state(self):
        tmp = self.state_p.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1))
        os.replace(str(tmp), str(self.state_p))

    # ------------------------------------------------------------------ inputs
    def arms(self):
        rows = [l.rstrip("\n").split("\t") for l in Path(self.a.arms).read_text().splitlines() if l.strip()]
        return [dict(zip(rows[0], r)) for r in rows[1:]]

    def verdicts(self):
        out = {}
        for p in sorted(self.vdir.glob("verdict_*.json")):
            fm = p.name[len("verdict_"):-len(".json")]
            try:
                d = json.loads(p.read_text())
            except Exception as e:
                out[fm] = dict(status="UNREADABLE", _sha=None, _err=repr(e)); continue
            d["status"] = str(d.get(self.a.verdict_key, "PENDING")).upper()
            d["_sha"] = sha(p)
            out[fm] = d
        return out

    def jobs(self, name):
        rc, o, e = run(["squeue", "-h", "-u", self.user, "-n", name, "-o", "%i|%T|%y|%r|%b"])
        res = []
        for l in o.decode().splitlines():
            p = l.strip().split("|")
            if len(p) >= 3:
                g = p[4] if len(p) > 4 else ""
                try:
                    ngpu = int(g.rsplit(":", 1)[-1]) if "gpu" in g else 0
                except ValueError:
                    ngpu = 1
                res.append(dict(id=p[0], state=p[1], nice=p[2], reason=p[3] if len(p) > 3 else "", ngpu=ngpu))
        return res

    # ------------------------------------------------------------------ hub
    def ps(self, script):
        enc = base64.b64encode(script.encode("utf-16-le")).decode()
        return run(self.ssh + ["powershell", "-NoProfile", "-EncodedCommand", enc], timeout=120)

    def race_owners(self):
        rc, o, e = self.ps("$d='%s/race'; if (Test-Path $d) { Get-ChildItem $d -Directory | ForEach-Object { $f=Join-Path $_.FullName 'owner.txt'; "
                           "if (Test-Path $f) { $_.Name + ' ' + (Get-Content $f -TotalCount 1) } } }" % HUB_WIN)
        if rc != 0:
            return None
        own = {}
        for l in o.decode(errors="replace").splitlines():
            p = l.strip().split()
            if len(p) >= 2:
                own[p[0]] = p[1]
        return own

    def push_hub(self, rels, base=None):
        rc, o, e = run(["tar", "-cf", "-", "-C", str(base or self.xs)] + rels, timeout=60)
        if rc != 0:
            return rc, e
        rc2, o2, e2 = run(self.ssh + ["tar -xf - -C %s" % HUB_WIN], inp=o, timeout=120)
        return rc2, e2

    def pull_hub(self, rel, dest):
        rc, o, e = run(self.ssh + ["tar -cf - -C %s %s" % (HUB_WIN, rel)], timeout=120)
        if rc != 0:
            return rc, e
        dest.mkdir(parents=True, exist_ok=True)
        rc2, o2, e2 = run(["tar", "-xf", "-", "-C", str(dest)], inp=o)
        return rc2, e2

    # ------------------------------------------------------------------ priority (A4 rule)
    def rank_key(self, arm, vd, init_rank):
        """Prereg A4.3(b) within a regime: controls (fm '-': NOFM, PEINIT) fixed at the top in arms.tsv init_rank order; FM arms
        by the verdict's priority_score (A3.6/A3.7 top-level field; lexicographic in status, sets_passed, m_ord, m_t0) descending,
        ties by fm id ascending. An FM with no verdict file yet counts as PENDING with score 2e9 (status_rank 2 only) and is
        ordered by its capacity-note init_rank among such FMs (first dispatch wave: NOFM, dinov3-vitl16, cleandift-sd21 or
        sd2.1-dift, PEINIT)."""
        fm = arm.get("fm", "-")
        if fm == "-":
            return (0, 0.0, init_rank, "")
        v = vd.get(fm)
        if v is None or v.get("priority_score") is None:
            return (1, -2e9, init_rank, fm)
        return (1, -float(v["priority_score"]), 0.0, fm)

    def priority(self, arms, vd):
        init = {a["arm_id"]: float(a.get("init_rank", 999)) for a in arms}
        live = [a for a in arms if not (a.get("fm", "-") != "-" and vd.get(a["fm"], {}).get("status") == "FAIL")]
        order = sorted(live, key=lambda a: (0 if a.get("regime") == "warm" else 1,) + self.rank_key(a, vd, init[a["arm_id"]]))
        nice = {a["arm_id"]: NICE_BASE + NICE_STEP * i for i, a in enumerate(order)}
        hold = [a["arm_id"] for a in live if a.get("hold_until_pass", "0") == "1" and not str(vd.get(a.get("fm", "-"), {}).get("status", "")).startswith("PASS")]
        return dict(utc=utc(), rule="warm before scratch; controls first (init_rank); FM arms by verdict priority_score desc (A3.6), ties fm id asc; no verdict file -> score 2e9 + capacity-note init_rank",
                    order=[a["arm_id"] for a in order], nice=nice, hold=hold,
                    verdicts={k: dict(status=v["status"], sha256=v["_sha"], priority_score=v.get("priority_score")) for k, v in vd.items()})

    # ------------------------------------------------------------------ one cycle
    def cycle(self):
        arms = self.arms()
        if self.site == "clusterB":
            tmp = self.adm / "hub_pull"
            rc, e = self.pull_hub("hub", tmp)
            if rc != 0:
                self.log("hub", "pull failed rc=%s %s" % (rc, e[-200:].decode(errors="replace") if e else ""))
                return
            if not self.install_from_hub(tmp / "hub"):
                return
        vd = self.verdicts()
        for fm, v in vd.items():
            if self.state["verdict_sha"].get(fm) != v["_sha"]:
                self.log("verdict", "verdict_%s.json status=%s sha256=%s" % (fm, v["status"], v["_sha"]), cand=True)
                self.state["verdict_sha"][fm] = v["_sha"]
        # gate cancellations
        for arm in arms:
            fm = arm.get("fm", "-")
            if fm == "-" or vd.get(fm, {}).get("status") != "FAIL":
                continue
            js = self.jobs("fx_%s" % arm["arm_id"])
            for j in js:
                if self.a.dry_run:
                    self.log("gate", "DRY-RUN would scancel %s fx_%s (%s): FM %s FAIL" % (j["id"], arm["arm_id"], j["state"], fm)); continue
                rc, o, e = run(["scancel", j["id"]])
                self.log("gate", "scancel %s fx_%s state=%s rc=%s: FM %s verdict FAIL (verdict sha256 %s)" % (j["id"], arm["arm_id"], j["state"], rc, fm, vd[fm]["_sha"]), cand=True)
                with open(str(self.adm / "cancellation_log.md"), "a") as f:
                    f.write("- %s scancel %s fx_%s (%s) site=%s: FM %s verdict FAIL, verdict sha256 %s\n" % (utc(), j["id"], arm["arm_id"], j["state"], self.site, fm, vd[fm]["_sha"]))
            if self.site == "clusterA" and not self.a.dry_run and "vps" in arm.get("lanes", ""):
                self.vps_remove_queued(arm["arm_id"])
            if arm["arm_id"] not in self.state["cancelled"] and not self.a.dry_run:
                rd = self.xs / "runs" / ("%s_%s" % (arm["arm_id"], self.site))
                rd.mkdir(parents=True, exist_ok=True)
                (rd / "CANCELLED_GATE_FAIL.json").write_text(json.dumps(dict(arm=arm["arm_id"], fm=fm, site=self.site, jobs=js, verdict_sha256=vd[fm]["_sha"], utc=utc())))
                self.state["cancelled"][arm["arm_id"]] = utc()
        # races
        own = self.race_owners()
        if own is None:
            self.log("race", "race owner listing failed (VPS unreachable?)")
        else:
            for arm in arms:
                o = own.get(arm["arm_id"])
                if o is None or o == self.site:
                    continue
                for j in self.jobs("fx_%s" % arm["arm_id"]):
                    if j["state"] in ("PENDING", "RUNNING", "CONFIGURING", "REQUEUED") and not self.a.dry_run:
                        rc, _, _ = run(["scancel", j["id"]])
                        self.log("race", "scancel %s fx_%s (%s) on %s: race lost, the %s copy started first; rc=%s" % (j["id"], arm["arm_id"], j["state"], self.site, o, rc), cand=True)
        # priority (CLUSTER-A computes; CLUSTER-B applies the hub copy of priority.json so both clusters use one ranking)
        if self.site == "clusterA":
            pr = self.priority(arms, vd)
            (self.adm / "priority.json").write_text(json.dumps(pr, indent=1))
        else:
            pp = self.adm / "hub_pull" / "hub" / "priority.json"
            pr = json.loads(pp.read_text()) if pp.exists() else self.priority(arms, vd)
        self.apply_priority(arms, pr)
        self.relay_targets(arms, vd)
        if self.site == "clusterA":
            self.update_queue(arms, vd)
        if self.site == "clusterA":
            self.publish()
            self.merge_clusterB_log()
        self.save_state()

    def needs_ok(self, arm):
        n = arm.get("needs", "-")
        if n in ("-", ""):
            return True
        if n.startswith("extractor:"):
            return False
        return Path(n.replace("$U", str(self.root))).exists()

    def apply_priority(self, arms, pr):
        """nice = rank order; GPU cap (A4.3: CLUSTER-A <= --cap GPUs for this track, counted from the jobs' gres): at most cap - running GPUs are left eligible
        (pending, not held), chosen in priority order among arms whose inputs exist ('needs'); every other pending fx_ job is
        held. Held jobs are released by the same rule when a slot frees or their inputs appear."""
        byid = {a["arm_id"]: a for a in arms}
        jobs = {aid: self.jobs("fx_%s" % aid) for aid in pr["order"]}
        running = sum(j["ngpu"] for js in jobs.values() for j in js if j["state"] in ("RUNNING", "CONFIGURING", "COMPLETING"))
        slots = max(0, self.a.cap - running)
        for aid in pr["order"]:
            nice = pr["nice"][aid]
            for j in jobs[aid]:
                if j["state"] != "PENDING":
                    continue
                if str(j["nice"]) != str(nice) and not self.a.dry_run:
                    rc, _, e = run(["scontrol", "update", "jobid=%s" % j["id"], "nice=%d" % nice])
                    self.log("priority", "nice %s -> %d for %s fx_%s rc=%s %s" % (j["nice"], nice, j["id"], aid, rc, e.decode(errors="replace").strip()[:120]))
                held = "JobHeldUser" in (j.get("reason") or "") or "JobHeldAdmin" in (j.get("reason") or "")
                want_eligible = slots >= max(1, j["ngpu"]) and self.needs_ok(byid[aid]) and aid not in pr["hold"]
                if want_eligible:
                    slots -= max(1, j["ngpu"])
                    if held and not self.a.dry_run:
                        rc, _, _ = run(["scontrol", "release", j["id"]])
                        self.log("priority", "release %s fx_%s rc=%s (slot free, inputs present, rank %d)" % (j["id"], aid, rc, pr["order"].index(aid) + 1))
                elif not held and not self.a.dry_run:
                    why = "inputs missing (%s)" % byid[aid].get("needs") if not self.needs_ok(byid[aid]) else "GPU cap %d reached" % self.a.cap
                    rc, _, _ = run(["scontrol", "hold", j["id"]])
                    self.log("priority", "hold %s fx_%s rc=%s: %s" % (j["id"], aid, rc, why))

    def update_queue(self, arms, vd):
        import csv
        q = self.adm / "arm_queue.csv"
        if not q.exists():
            return
        rows = list(csv.DictReader(open(str(q))))
        cols = list(rows[0].keys()) if rows else []
        changed = False
        for r in rows:
            aid = r["arm_id"]
            rd = self.xs / "runs" / ("%s_clusterA" % aid)
            st = r["state"]
            js = self.jobs("fx_%s" % aid)
            if (rd / "GATE_FAIL").exists() or (rd / "CANCELLED_GATE_FAIL.json").exists() or (r["fm"] != "-" and vd.get(r["fm"], {}).get("status") == "FAIL"):
                new = "CANCELLED_GATE_FAIL"
            elif (rd / "eval" / "best_milestone.txt").exists():
                new = "DONE"
            elif any(j["state"] in ("RUNNING", "CONFIGURING", "COMPLETING") for j in js):
                new = "RUNNING"
            elif js:
                new = "SUBMITTED"
            else:
                new = st
            if new != st:
                r["state"] = new; r["updated_utc"] = utc(); changed = True
                log = "cancellation_log.md" if new == "CANCELLED_GATE_FAIL" else "arm_log.md"
                with open(str(self.adm / log), "a") as f:
                    f.write("- %s %s: %s -> %s (controller, jobs %s)\n" % (utc(), aid, st, new, ",".join(j["id"] for j in js) or "none"))
        if changed:
            tmp = q.with_suffix(".tmp")
            with open(str(tmp), "w", newline="") as fh:
                w = csv.DictWriter(fh, cols); w.writeheader(); w.writerows(rows)
            os.replace(str(tmp), str(q))

    def vps_remove_queued(self, arm_id):
        """A4.3(a): remove queued (not running) VPS runner cmds 400-499 of a FAIL FM's arm."""
        rc, o, e = self.ps("Get-ChildItem $VPS_ROOT/uhwr/runner/queue -Filter '4??_fx_%s.cmd' | ForEach-Object { Remove-Item $_.FullName; $_.Name }" % arm_id)
        names = o.decode(errors="replace").split()
        for nm in names:
            self.log("gate", "removed queued VPS cmd %s (arm %s, FM verdict FAIL)" % (nm, arm_id), cand=True)

    def relay_targets(self, arms, vd):
        """REPA target files to CLUSTER-B through the hub (background, sha-checked by hub_push/hub_pull manifests).
        CLUSTER-A: push exp/FMD/xs/targets/repa/<fm>/ once its DONE.json exists (never DINOv3-family FMs: licence lanes CLUSTER-A/VPS;
        never FAIL FMs); writes hub/targets_ready.txt. CLUSTER-B: pull every FM listed there that has a CLUSTER-B-lane arm alive."""
        jobs_dir = self.adm / "relay"; jobs_dir.mkdir(exist_ok=True)
        st = self.state.setdefault("relay", {})
        for f, p in list(st.items()):                       # reap finished background transfers
            if p.get("pid") and p.get("rc") is None:
                try:
                    pid, status = os.waitpid(p["pid"], os.WNOHANG)
                except ChildProcessError:
                    pid, status = p["pid"], 0
                if pid:
                    p["rc"] = os.waitstatus_to_exitcode(status) if hasattr(os, "waitstatus_to_exitcode") else status
                    self.log("relay", "%s targets %s rc=%s (log %s)" % ("push" if self.site == "clusterA" else "pull", f, p["rc"], p["log"]))
        fms = sorted({a["fm"] for a in arms if a.get("fm", "-") != "-" and "clusterB" in a.get("lanes", "") and a.get("repa_targets", "-") != "-"})
        busy = sum(1 for p in st.values() if p.get("pid") and p.get("rc") is None)
        bb = self.xs / "code/fmdx_backbone_v1/jobs"
        if self.site == "clusterA":
            ready = sorted(f for f, p in st.items() if p.get("rc") == 0)
            (self.xs / "hub").mkdir(exist_ok=True)
            (self.xs / "hub" / "targets_ready.txt").write_text("".join(f + "\n" for f in ready))
            for f in fms:
                if f in st or vd.get(f, {}).get("status") == "FAIL" or busy >= 1:
                    continue
                if not list((self.xs / "targets/repa" / f).glob("*.DONE.json")):
                    continue
                log = jobs_dir / ("push_%s.log" % f)
                files = sorted(str(q.relative_to(self.root)) for q in (self.xs / "targets/repa" / f).iterdir() if q.is_file())
                pr = subprocess.Popen(["bash", str(bb / "hub_push.sh"), "tgt_%s" % f] + files,
                                      cwd=str(self.root), stdout=open(str(log), "a"), stderr=subprocess.STDOUT)
                st[f] = dict(pid=pr.pid, rc=None, log=str(log), started=utc()); busy += 1
                self.log("relay", "push targets %s started pid %s" % (f, pr.pid))
        else:
            rp = self.adm / "hub_pull" / "hub" / "targets_ready.txt"
            ready = [l.strip() for l in rp.read_text().splitlines() if l.strip()] if rp.exists() else []
            for f in ready:
                if f in st or f not in fms or busy >= 1 or vd.get(f, {}).get("status") == "FAIL":
                    continue
                log = jobs_dir / ("pull_%s.log" % f)
                pr = subprocess.Popen(["bash", str(bb / "hub_pull.sh"), "tgt_%s" % f, "exp/FMD/xs/targets/repa/%s" % f],
                                      cwd=str(self.root), stdout=open(str(log), "a"), stderr=subprocess.STDOUT)
                st[f] = dict(pid=pr.pid, rc=None, log=str(log), started=utc()); busy += 1
                self.log("relay", "pull targets %s started pid %s" % (f, pr.pid))

    def publish(self):
        hub = self.xs / "hub"
        (hub / "verdicts").mkdir(parents=True, exist_ok=True)
        files = []
        for p in sorted(self.vdir.glob("verdict_*.json")):
            q = hub / "verdicts" / p.name
            q.write_bytes(p.read_bytes()); files.append(q)
        for src in (Path(self.a.arms), self.adm / "priority.json"):
            q = hub / src.name; q.write_bytes(src.read_bytes()); files.append(q)
        (hub / "SHA256SUMS_hub.txt").write_text("".join("%s  %s\n" % (sha(q), q.relative_to(hub).as_posix()) for q in files))
        rc, e = self.push_hub(["hub"])
        if rc != 0:
            self.log("hub", "push failed rc=%s %s" % (rc, e[-200:].decode(errors="replace") if e else ""))

    def install_from_hub(self, hub):
        sums = hub / "SHA256SUMS_hub.txt"
        if not sums.exists():
            self.log("hub", "no SHA256SUMS_hub.txt in the hub copy"); return False
        entries = [l.split("  ", 1) for l in sums.read_text().splitlines() if l.strip()]
        bad = [rel for h, rel in entries if not (hub / rel).exists() or sha(hub / rel) != h]
        if bad:
            self.log("hub", "sha256 mismatch in hub copy %s: nothing applied" % bad[:5]); return False
        self.vdir.mkdir(parents=True, exist_ok=True)
        for h, rel in entries:
            if rel.startswith("verdicts/"):
                dst = self.vdir / rel.split("/", 1)[1]
                if not dst.exists() or sha(dst) != h:
                    tmp = dst.with_suffix(".tmp"); tmp.write_bytes((hub / rel).read_bytes()); os.replace(str(tmp), str(dst))
                    self.log("verdict", "installed %s sha256=%s from hub" % (dst.name, h))
            elif rel == "arms.tsv" and (not Path(self.a.arms).exists() or sha(self.a.arms) != h):
                tmp = Path(self.a.arms).with_suffix(".tmp"); tmp.write_bytes((hub / rel).read_bytes()); os.replace(str(tmp), str(self.a.arms))
                self.log("hub", "installed arms.tsv sha256=%s from hub (CLUSTER-A is the only writer)" % h)
        rc, o, e = run(["tar", "-cf", "-", "-C", str(self.adm), self.log_path.name])
        if rc == 0:
            run(self.ssh + ["powershell", "-NoProfile", "-EncodedCommand",
                            base64.b64encode(("New-Item -ItemType Directory -Force -Path '%s/hub_clusterB' | Out-Null" % HUB_WIN).encode("utf-16-le")).decode()])
            run(self.ssh + ["tar -xf - -C %s/hub_clusterB" % HUB_WIN], inp=o)
        return True

    def merge_clusterB_log(self):
        tmp = self.adm / "hub_clusterB_pull"
        rc, e = self.pull_hub("hub_clusterB", tmp)
        p = tmp / "hub_clusterB" / "controller_log_clusterB.md"
        if rc != 0 or not p.exists():
            return
        seen = set(self.state["seen_clusterB_lines"])
        for l in p.read_text().splitlines():
            if "(id " not in l:
                continue
            lid = l.rsplit("(id ", 1)[1].rstrip(")")
            if lid in seen:
                continue
            with open(str(self.log_path), "a") as f:
                f.write(l + "  [merged from CLUSTER-B]\n")
            if "[gate]" in l or "[race]" in l or "[verdict]" in l:
                with open(str(self.cand), "a") as f:
                    f.write("- FMDX controller [clusterB, merged] %s\n" % l[2:])
            seen.add(lid)
        self.state["seen_clusterB_lines"] = sorted(seen)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", choices=("clusterA", "clusterB"), required=True)
    ap.add_argument("--root", required=True, help="CLUSTER-A $U ; CLUSTER-B $U_B")
    ap.add_argument("--arms", required=True)
    ap.add_argument("--verdict-key", default="status")
    ap.add_argument("--cap", type=int, default=4, help="max CLUSTER-A GPUs (1 per fx_ job) running or eligible for this track")
    ap.add_argument("--period", type=int, default=300)
    ap.add_argument("--until", default="2026-11-10T23:59:00Z")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    c = Ctl(a)
    c.log("start", "controller start site=%s pid=%s slurm=%s host=%s script_sha256=%s arms_sha256=%s dry_run=%s" % (
        a.site, os.getpid(), os.environ.get("SLURM_JOB_ID"), os.uname()[1], sha(__file__), sha(a.arms), a.dry_run))
    import calendar
    until = calendar.timegm(time.strptime(a.until, "%Y-%m-%dT%H:%M:%SZ"))
    while True:
        try:
            c.cycle()
        except Exception as e:
            c.log("error", "cycle error %r" % (e,))
        if a.once or time.time() > until:
            break
        time.sleep(a.period)
    c.log("stop", "controller stop")


if __name__ == "__main__":
    main()
