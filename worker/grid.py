#!/usr/bin/env python3
"""Adaptive ATR grid bot - paper trading on Hyperliquid public data (no orders are ever sent).
Two bots run side by side on the same 8 coins and the same feed:
  adaptive : validated config. Ladder re-centered on every fill (inventory skew, geometric adds).
  static   : hybrid lattice (6 buy + 6 sell rungs, uniform ATR steps), re-anchors when flat/stopped or when price exits the outer rung + 4 steps.
Both: spacing = K x ATR(24, 1h), Donchian(48) trend gate, basket z-gate, 2% stop on avg entry,
depth stop, BTC hedge, net-exposure cap. Fills are simulated on 5m candles (pessimistic: price must
trade PEN through the level, fees on every fill)."""
import json, math, os, sys, time, bisect, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.join(ROOT, "state", "state.json")
HL = "https://api.hyperliquid.xyz/info"
COINS = ["BTC", "ETH", "BNB", "LTC", "ADA", "XRP", "LINK", "ETC"]
START = 100000.0
LEV = float(os.environ.get("GRID_LEV", "3"))      # size multiple vs the 5%-per-level base book
Q0 = 0.05 * LEV                                    # notional per level as fraction of START equity
K, LMAX, G, BETA = 4.0, 6, 1.5, 1.0
SLACK = 4.0  # hybrid lattice: re-anchor when close leaves outer rung + SLACK steps
STALE_MS = 24 * 3600000
MINOFF = 0.25
SL, STOPN = 0.02, 4.0
THR, ZTHR = 1.0, 0.75
CAP = 0.8 * LEV
HEDGE = 1.0
FEE_M, FEE_T = 0.0002, 0.00045
PEN = 0.0002                                       # limit must trade 2bp THROUGH the level
STOP_SLIP = 0.0005
BAR = 300000

def hl(p, tries=3):
    for a in range(tries):
        try:
            rq = urllib.request.Request(HL, data=json.dumps(p).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(rq, timeout=30) as r: return json.loads(r.read())
        except Exception:
            if a == tries - 1: raise
            time.sleep(2 * (a + 1))

def candles(coin, tf, start, end):
    out = hl({"type": "candleSnapshot", "req": {"coin": coin, "interval": tf, "startTime": int(start), "endTime": int(end)}})
    return [{"t": c["t"], "o": float(c["o"]), "h": float(c["h"]), "l": float(c["l"]), "c": float(c["c"])} for c in out]

# ---------------- features (1h) ----------------
def features(h1):
    """h1: {coin: [bars]} aligned on common open times. returns (times, {coin: {A,TR,Z}}) per closed 1h bar."""
    common = None
    for c in COINS:
        s = {b["t"] for b in h1[c]}; common = s if common is None else common & s
    ts = sorted(common); idx = {c: {b["t"]: b for b in h1[c]} for c in COINS}
    n = len(ts); F = {c: {"A": [0.0] * n, "TR": [0.0] * n, "Z": [0.0] * n} for c in COINS}
    lp = {c: [math.log(idx[c][t]["c"]) for t in ts] for c in COINS}
    for c in COINS:
        a = None; pc = None
        for i, t in enumerate(ts):
            b = idx[c][t]
            tr = (b["h"] - b["l"]) if pc is None else max(b["h"] - b["l"], abs(b["h"] - pc), abs(b["l"] - pc))
            a = tr if a is None else a + (tr - a) / 24.0
            pc = b["c"]; F[c]["A"][i] = a
            w = [idx[c][ts[j]]["c"] for j in range(max(0, i - 47), i + 1)]
            hi, lo = max(w), min(w)
            F[c]["TR"][i] = (b["c"] - (hi + lo) / 2) / ((hi - lo) / 2 + 1e-12) * 2 if i >= 1 else 0.0
    sp = {c: [lp[c][i] - sum(lp[k][i] for k in COINS) / len(COINS) for i in range(n)] for c in COINS}
    for c in COINS:
        for i in range(n):
            w = sp[c][max(0, i - 167): i + 1]
            if len(w) >= 42:
                m = sum(w) / len(w); sd = math.sqrt(sum((x - m) ** 2 for x in w) / (len(w) - 1)) + 1e-9
                F[c]["Z"][i] = (sp[c][i] - m) / sd
    return ts, F

# ---------------- ladders ----------------
def gs(base, k, g):
    return sum(g ** m for m in range(base, base + k + 1))

def ladder_adaptive(px, s, d):
    cc = px - BETA * (d / LMAX) * s; buys = []; sells = []
    for r in range(LMAX - d):
        if d < 0 and r < -d: p = cc - s * (r + 1)
        else:
            k = r - (-d if d < 0 else 0); base = d if d > 0 else 0
            p = cc - (s * (-d) if d < 0 else 0.0) - s * gs(base, k, G)
        if r == 0 and p > px - MINOFF * s: p = px - MINOFF * s
        buys.append(p)
    for r in range(LMAX + d):
        if d > 0 and r < d: p = cc + s * (r + 1)
        else:
            k = r - (d if d > 0 else 0); base = -d if d < 0 else 0
            p = cc + (s * d if d > 0 else 0.0) + s * gs(base, k, G)
        if r == 0 and p < px + MINOFF * s: p = px + MINOFF * s
        sells.append(p)
    return buys, sells

def ladder_static(P0, s0, d):
    return ([P0 - (d + 1 + r) * s0 for r in range(LMAX - d)], [P0 - (d - 1 - r) * s0 for r in range(LMAX + d)])

# ---------------- bot ----------------
def new_coin():
    return {"pos": 0.0, "d": 0, "basis": 0.0, "cj": 0.0, "buys": [], "sells": [], "qty": 0.0, "anchor": 0.0, "aspace": 0.0,
            "lastfill": 0, "lastpx": 0.0, "lasts": 0.0, "ep_v0": None, "ep_R": 0.0, "ep_t": 0, "gated": [False, False]}

def new_bot(kind):
    return {"kind": kind, "coins": {c: new_coin() for c in COINS}, "hp": 0.0, "hcash": 0.0, "peak": START, "maxdd": 0.0,
            "equity": START, "fills": [], "episodes": [], "curve": [], "feepaid": 0.0, "nfills": 0, "nstops": 0}

def place(bot, c, px, s, t):
    st = bot["coins"][c]; d = st["d"]
    if bot["kind"] == "static":
        if d == 0: st["anchor"], st["aspace"] = px, s
        b, sl = ladder_static(st["anchor"], st["aspace"], d)
    else:
        b, sl = ladder_adaptive(px, s, d)
    st["buys"], st["sells"], st["qty"] = b, sl, Q0 * START / px
    st["lastfill"], st["lastpx"], st["lasts"] = t, px, s

def log_fill(bot, t, c, side, px, qty, kind, d):
    bot["fills"].insert(0, {"t": t, "coin": c, "side": side, "px": px, "usd": round(px * qty, 2), "kind": kind, "d": d})
    del bot["fills"][600:]

def close_episode(bot, c, t, px, kind):
    st = bot["coins"][c]
    if st["ep_v0"] is None: return
    v = st["cj"] + st["pos"] * px
    pnl = v - st["ep_v0"]; r = pnl / st["ep_R"] if st["ep_R"] else 0.0
    bot["episodes"].insert(0, {"t0": st["ep_t"], "t": t, "coin": c, "pnl": round(pnl, 2), "R": round(r, 3), "kind": kind})
    del bot["episodes"][600:]; st["ep_v0"] = None

def mkt_close(bot, c, t, px, kind, slip):
    st = bot["coins"][c]
    if st["pos"] == 0: return
    fx = px * (1 - slip) if st["pos"] > 0 else px * (1 + slip)
    f = FEE_T * abs(st["pos"]) * fx
    side = "SELL" if st["pos"] > 0 else "BUY"
    log_fill(bot, t, c, side, fx, abs(st["pos"]), kind, 0)
    st["cj"] += st["pos"] * fx - f; bot["feepaid"] += f; bot["nfills"] += 1
    st["pos"] = 0.0; st["basis"] = 0.0; st["d"] = 0
    close_episode(bot, c, t, fx, kind)

def net_notional(bot, px):
    return sum(bot["coins"][c]["pos"] * px[c] for c in COINS)

def step_coin(bot, c, t, o, h, l, cl, A, TR, Z, px):
    st = bot["coins"][c]
    s = K * A
    if not st["buys"] and not st["sells"] or (st["d"] == 0 and t - st["lastfill"] > STALE_MS):
        st["d"] = 0
        place(bot, c, o, s, t)
    okb = TR >= -THR and Z <= ZTHR; oks = TR <= THR and Z >= -ZTHR
    st["gated"] = [not okb, not oks]
    pts = (o, l, h, cl) if cl >= o else (o, h, l, cl)
    for sg in range(3):
        a, b = pts[sg], pts[sg + 1]
        for _ in range(60):
            d = st["d"]; best = None
            nn = net_notional(bot, px)
            if b < a:
                for r, p in enumerate(st["buys"]):
                    if b <= p * (1 - PEN) or (p > a and b <= a):
                        isadd = d >= 0 or r >= -d
                        if isadd and not okb: continue
                        if isadd and abs(nn + st["qty"] * p) > CAP * START and abs(nn + st["qty"] * p) > abs(nn): continue
                        if best is None or p > best[1]: best = (r, p, 1, isadd)
            else:
                for r, p in enumerate(st["sells"]):
                    if b >= p * (1 + PEN) or (p < a and b >= a):
                        isadd = d <= 0 or r >= d
                        if isadd and not oks: continue
                        if isadd and abs(nn - st["qty"] * p) > CAP * START and abs(nn - st["qty"] * p) > abs(nn): continue
                        if best is None or p < best[1]: best = (r, p, -1, isadd)
            if best is None: break
            r, p, side, isadd = best
            fx = min(p, a) if side == 1 else max(p, a)
            fee = (FEE_M if fx == p else FEE_T) * st["qty"] * fx
            qty = st["qty"]; pold = st["pos"]
            st["pos"] += side * qty; st["cj"] -= side * qty * fx + fee
            bot["feepaid"] += fee; bot["nfills"] += 1
            px[c] = fx
            if isadd: st["basis"] += side * qty * fx
            elif abs(pold) > 1e-15: st["basis"] *= st["pos"] / pold
            st["d"] += side
            if st["d"] == 0 and abs(st["pos"]) > 1e-12: st["cj"] += st["pos"] * fx; st["pos"] = 0.0
            if st["d"] == 0: st["basis"] = 0.0
            log_fill(bot, t, c, "BUY" if side == 1 else "SELL", fx, qty, "fill" if fx == p else "gate-reopen", st["d"])
            if st["d"] == 0: close_episode(bot, c, t, fx, "tp")
            elif st["ep_v0"] is None:
                st["ep_v0"] = st["cj"] + st["pos"] * fx + fee; st["ep_t"] = t
                st["ep_R"] = Q0 * START * (K * A / fx)
            place(bot, c, fx, s, t); a = fx
    # hybrid re-anchor (static lattice only): price left the lattice range
    if bot["kind"] == "static" and st["aspace"] > 0:
        rng = (LMAX + SLACK) * st["aspace"]
        if cl < st["anchor"] - rng or cl > st["anchor"] + rng:
            st["aspace"] = s; st["anchor"] = cl + st["d"] * s
            b2, s2 = ladder_static(st["anchor"], st["aspace"], st["d"])
            st["buys"], st["sells"], st["qty"] = b2, s2, Q0 * START / cl
            st["lastfill"], st["lastpx"], st["lasts"] = t, cl, s
    # stops (bar level)
    d = st["d"]
    if d >= LMAX or d <= -LMAX:
        stp = st["lastpx"] - STOPN * st["lasts"] if d > 0 else st["lastpx"] + STOPN * st["lasts"]
        if (d > 0 and l <= stp) or (d < 0 and h >= stp):
            mkt_close(bot, c, t, stp, "depth-stop", STOP_SLIP); bot["nstops"] += 1; place(bot, c, stp, s, t)
    if st["pos"] != 0 and st["basis"] != 0:
        av = st["basis"] / st["pos"]; stp = av * (1 - SL) if st["pos"] > 0 else av * (1 + SL)
        if (st["pos"] > 0 and l <= stp) or (st["pos"] < 0 and h >= stp):
            mkt_close(bot, c, t, stp, "pct-stop", STOP_SLIP); bot["nstops"] += 1; place(bot, c, stp, s, t)
    px[c] = cl

def end_bar(bot, t, cl):
    nt = net_notional(bot, cl)
    tgt = -HEDGE * nt / cl["BTC"]
    if abs(tgt - bot["hp"]) * cl["BTC"] > 0.05 * START:
        dif = tgt - bot["hp"]; f = FEE_T * abs(dif) * cl["BTC"]
        bot["hcash"] -= dif * cl["BTC"] + f; bot["feepaid"] += f; bot["hp"] = tgt
    eq = START + bot["hcash"] + bot["hp"] * cl["BTC"] + sum(bot["coins"][c]["cj"] + bot["coins"][c]["pos"] * cl[c] for c in COINS)
    bot["equity"] = eq; bot["peak"] = max(bot["peak"], eq)
    bot["maxdd"] = max(bot["maxdd"], (bot["peak"] - eq) / bot["peak"])
    if t % 3600000 == 0 or not bot["curve"]:
        bot["curve"].append([t, round(eq, 2)])
    elif t - bot["curve"][-1][0] < 3600000: bot["curve"][-1] = [t, round(eq, 2)] if bot["curve"][-1][0] % 3600000 else bot["curve"][-1]

def run(f5, ts1, F, bots, t_from):
    """f5: {coin: {t: bar}} 5m bars; process every common 5m bar > t_from in order."""
    common = None
    for c in COINS:
        s = set(f5[c]); common = s if common is None else common & s
    times = sorted(t for t in common if t > t_from)
    for t in times:
        j = bisect.bisect_right(ts1, t - 3600000) - 1   # last closed 1h bar (open + 1h <= t)
        if j < 1: continue
        px = {c: f5[c][t]["o"] for c in COINS}
        for bot in bots.values():
            pxb = dict(px)
            for c in COINS:
                b = f5[c][t]
                step_coin(bot, c, t, b["o"], b["h"], b["l"], b["c"], F[c]["A"][j], F[c]["TR"][j], F[c]["Z"][j], pxb)
            end_bar(bot, t, {c: f5[c][t]["c"] for c in COINS})
    return times[-1] if times else t_from

def summarize(bot, last_close):
    pos = []
    for c in COINS:
        st = bot["coins"][c]
        if st["pos"] != 0:
            av = st["basis"] / st["pos"] if st["basis"] else 0
            pos.append({"coin": c, "qty": st["pos"], "usd": round(st["pos"] * last_close[c], 2), "avg": av, "d": st["d"],
                        "upnl": round(st["pos"] * (last_close[c] - av), 2) if av else 0})
    orders = []
    for c in COINS:
        st = bot["coins"][c]
        for p in st["buys"]: orders.append({"coin": c, "side": "BUY", "px": p, "usd": round(st["qty"] * p, 2), "paused": st["gated"][0] and (st["d"] >= 0 or False)})
        for p in st["sells"]: orders.append({"coin": c, "side": "SELL", "px": p, "usd": round(st["qty"] * p, 2), "paused": st["gated"][1] and (st["d"] <= 0 or False)})
    eps = bot["episodes"]; n = len(eps)
    wins = [e for e in eps if e["pnl"] > 0]
    return {"positions": pos, "orders": orders, "hedge": {"qty": bot["hp"], "usd": round(bot["hp"] * last_close["BTC"], 2)},
            "stats": {"episodes": n, "win": round(len(wins) / n, 3) if n else None, "evR": round(sum(e["R"] for e in eps) / n, 3) if n else None,
                      "fills": bot["nfills"], "fees": round(bot["feepaid"], 2), "stops": bot["nstops"]}}

def main():
    now = int(time.time() * 1000); now_bar = now // BAR * BAR   # first non-closed bar open
    h1 = {c: candles(c, "1h", now - 700 * 3600000, now) for c in COINS}
    ts1, F = features(h1)
    if os.path.exists(STATE): S = json.load(open(STATE))
    else: S = {"started": now, "last_ts": 0, "config": {}, "bots": {"adaptive": new_bot("adaptive"), "static": new_bot("static")}}
    S["config"] = {"coins": COINS, "lev": LEV, "k": K, "levels": LMAX, "geo": G, "atr": 24, "donchian": 48, "sl": SL, "pen_bp": PEN * 1e4,
                   "maker_bp": FEE_M * 1e4, "taker_bp": FEE_T * 1e4, "hedge": HEDGE, "start": START}
    t_from = S["last_ts"]
    start_fetch = t_from + BAR if t_from else now - 3 * BAR
    f5 = {}
    for c in COINS:
        f5[c] = {b["t"]: b for b in candles(c, "5m", max(start_fetch, now - 4900 * BAR), now) if b["t"] < now_bar}
    if not t_from:   # first run: begin from the last closed bar open (flat, orders centered there)
        t_from = max(min(max(f5[c]) for c in COINS) - BAR, 0)
    last = run(f5, ts1, F, S["bots"], t_from)
    S["last_ts"] = last; S["updated"] = now
    lc = {c: (f5[c][last]["c"] if last in f5[c] else 0) for c in COINS}
    S["prices"] = lc
    S["summary"] = {k: summarize(b, lc) for k, b in S["bots"].items()}
    S["features"] = {c: {"atr": F[c]["A"][-1], "trend": round(F[c]["TR"][-1], 3), "z": round(F[c]["Z"][-1], 3)} for c in COINS}
    json.dump(S, open(STATE, "w"), separators=(",", ":"))
    for k, b in S["bots"].items(): print(k, round(b["equity"], 2), "dd", round(b["maxdd"] * 100, 2), "%", "fills", b["nfills"], "eps", len(b["episodes"]))

if __name__ == "__main__":
    main()
