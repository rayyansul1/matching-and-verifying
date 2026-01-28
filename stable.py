from __future__ import annotations
import sys, time, random, csv, os
from typing import List, Tuple

def read_prefs(path: str) -> tuple[int, List[List[int]], List[List[int]]]:
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            tokens = [t for t in f.read().split()]
    except FileNotFoundError:
        raise ValueError(f"Could not open input file: {path}")

    if not tokens:
        raise ValueError("Empty input file.")

    it = iter(tokens)
    try:
        n = int(next(it))
    except StopIteration:
        raise ValueError("Empty input file.")
    if n <= 0:
        raise ValueError("n must be positive.")

    def read_block(kind: str) -> List[List[int]]:
        block: List[List[int]] = []
        for r in range(n):
            row = []
            for _ in range(n):
                try:
                    row.append(int(next(it)))
                except StopIteration:
                    raise ValueError(f"Not enough numbers for {kind} preferences.")
            row0 = [x - 1 for x in row]
            if sorted(row0) != list(range(n)):
                raise ValueError(f"{kind} preference line {r+1} is not a permutation of 1..n.")
            block.append(row0)
        return block

    hospitals = read_block("hospital")
    students = read_block("student")
    return n, hospitals, students

def read_matching(path: str) -> List[Tuple[int,int]]:
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    except FileNotFoundError:
        raise ValueError(f"Could not open matching file: {path}")

    pairs: List[Tuple[int,int]] = []
    for ln in lines:
        parts = ln.split()
        if len(parts) != 2:
            raise ValueError(f"Bad matching line: '{ln}' (expected two integers)")
        pairs.append((int(parts[0]), int(parts[1])))
    return pairs

def print_matching(match_h: List[int]) -> None:
    for h, s in enumerate(match_h):
        print(f"{h+1} {s+1}")

def gale_shapley(n: int, hospitals: List[List[int]], students: List[List[int]]) -> Tuple[List[int], int]:
    rank_s = [[0]*n for _ in range(n)]
    for s in range(n):
        for r, h in enumerate(students[s]):
            rank_s[s][h] = r

    match_s = [-1]*n
    match_h = [-1]*n
    next_choice = [0]*n
    free = list(range(n))
    proposals = 0

    while free:
        h = free.pop()
        if next_choice[h] >= n:
            continue
        s = hospitals[h][next_choice[h]]
        next_choice[h] += 1
        proposals += 1

        cur = match_s[s]
        if cur == -1:
            match_s[s] = h
            match_h[h] = s
        else:
            if rank_s[s][h] < rank_s[s][cur]:
                match_s[s] = h
                match_h[h] = s
                match_h[cur] = -1
                free.append(cur)
            else:
                if next_choice[h] < n:
                    free.append(h)

    return match_h, proposals

def verify(n: int, hospitals: List[List[int]], students: List[List[int]], pairs_1based: List[Tuple[int,int]]) -> str:
    # Validity checks
    if len(pairs_1based) != n:
        return f"INVALID: expected {n} lines, got {len(pairs_1based)}."

    match_h = [-1]*n
    seen_h, seen_s = set(), set()
    for (i1, j1) in pairs_1based:
        if not (1 <= i1 <= n and 1 <= j1 <= n):
            return f"INVALID: indices out of range in pair ({i1}, {j1})."
        h, s = i1-1, j1-1
        if h in seen_h:
            return f"INVALID: hospital {i1} appears more than once."
        if s in seen_s:
            return f"INVALID: student {j1} matched more than once."
        seen_h.add(h); seen_s.add(s)
        match_h[h] = s

    if any(x == -1 for x in match_h):
        return "INVALID: not every hospital has exactly one match."

    match_s = [-1]*n
    for h in range(n):
        match_s[match_h[h]] = h

    rank_h = [[0]*n for _ in range(n)]
    for h in range(n):
        for r, s in enumerate(hospitals[h]):
            rank_h[h][s] = r

    rank_s = [[0]*n for _ in range(n)]
    for s in range(n):
        for r, h in enumerate(students[s]):
            rank_s[s][h] = r

    for h in range(n):
        s_assigned = match_h[h]
        for s in range(n):
            if s == s_assigned:
                continue
            if rank_h[h][s] < rank_h[h][s_assigned]:
                h_assigned = match_s[s]
                if rank_s[s][h] < rank_s[s][h_assigned]:
                    return f"UNSTABLE: blocking pair (hospital {h+1}, student {s+1})."

    return "VALID STABLE"

def rand_prefs(n: int, rng: random.Random) -> tuple[List[List[int]], List[List[int]]]:
    base = list(range(n))
    hos, stu = [], []
    for _ in range(n):
        x = base[:]
        rng.shuffle(x)
        hos.append(x)
    for _ in range(n):
        x = base[:]
        rng.shuffle(x)
        stu.append(x)
    return hos, stu

def benchmark(trials: int = 5) -> None:
    import matplotlib.pyplot as plt

    sizes = [1,2,4,8,16,32,64,128,256,512]
    rng = random.Random(0)

    m_times, v_times = [], []

    for n in sizes:
        m_sum = 0.0
        v_sum = 0.0
        for _ in range(trials):
            hos, stu = rand_prefs(n, rng)

            t0 = time.perf_counter()
            match_h, _ = gale_shapley(n, hos, stu)
            t1 = time.perf_counter()
            m_sum += (t1 - t0)

            pairs = [(h+1, match_h[h]+1) for h in range(n)]
            t2 = time.perf_counter()
            _ = verify(n, hos, stu, pairs)
            t3 = time.perf_counter()
            v_sum += (t3 - t2)

        m_times.append(m_sum / trials)
        v_times.append(v_sum / trials)

    os.makedirs("results", exist_ok=True)

    with open("results/timings.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n","matcher_seconds","verifier_seconds"])
        for i, n in enumerate(sizes):
            w.writerow([n, m_times[i], v_times[i]])

    plt.figure()
    plt.plot(sizes, m_times, marker="o")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("n (log2)")
    plt.ylabel("seconds (log)")
    plt.title("Matcher runtime vs n")
    plt.tight_layout()
    plt.savefig("results/matcher_runtime.png", dpi=200)

    plt.figure()
    plt.plot(sizes, v_times, marker="o")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("n (log2)")
    plt.ylabel("seconds (log)")
    plt.title("Verifier runtime vs n")
    plt.tight_layout()
    plt.savefig("results/verifier_runtime.png", dpi=200)

    print("Wrote results/timings.csv and results/*_runtime.png")

def usage() -> int:
    print(
        "Usage:\n"
        "  python stable.py match <input.in>            > match.out\n"
        "  python stable.py verify <input.in> <match.out>\n"
        "  python stable.py bench\n",
        file=sys.stderr
    )
    return 2

def main(argv: List[str]) -> int:
    if len(argv) < 2:
        return usage()

    mode = argv[1].lower()

    if mode == "match":
        if len(argv) != 3:
            return usage()
        try:
            n, hos, stu = read_prefs(argv[2])
            match_h, _ = gale_shapley(n, hos, stu)
            print_matching(match_h)
            return 0
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1

    if mode == "verify":
        if len(argv) != 4:
            return usage()
        try:
            n, hos, stu = read_prefs(argv[2])
            pairs = read_matching(argv[3])
            print(verify(n, hos, stu, pairs))
            return 0
        except ValueError as e:
            print(f"INVALID: {e}")
            return 0

    if mode == "bench":
        benchmark(trials=5)
        return 0

    return usage()

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

