import math
import random
import statistics

S = 336
SEED = 2026

def erlang_b(A, c):
    b = 1.0
    for j in range(1, c + 1):
        b = A * b / (j + A * b)
    return b

def capacity(c, target=0.02):
    lo, hi = 0.0, 2.0 * max(1, c)
    for _ in range(60):
        mid = (lo + hi) / 2
        if erlang_b(mid, c) > target:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2

def valid_clusters(limit=50):
    return sorted({
        i*i + i*j + j*j
        for i in range(limit)
        for j in range(limit)
        if 0 < i+j and i*i + i*j + j*j <= limit
    })

def minimum_N(n, k, threshold_db=15):
    return next(
        N for N in valid_clusters()
        if 10 * math.log10((3*N)**(n/2) / k) >= threshold_db
    )

def sample_sir(N, n, k, samples=15000,
               seed=SEED, shadow_db=0):
    rng = random.Random(seed)
    Q = math.sqrt(3*N)
    results = []

    for _ in range(samples):
        r = math.sqrt(rng.random())
        theta = 2 * math.pi * rng.random()
        x = r * math.cos(theta)
        y = r * math.sin(theta)

        desired = max(r, 1e-4)**(-n)
        interference = 0.0

        for j in range(k):
            a = 2 * math.pi * j / 6
            d = math.hypot(
                Q * math.cos(a) - x,
                Q * math.sin(a) - y
            )
            sh = rng.gauss(0, shadow_db)
            interference += d**(-n) * 10**(sh/10)

        sir = 10 * math.log10(desired / interference)
        results.append(sir)

    results.sort()

    return (
        statistics.mean(results),
        results[int(0.05 * samples)],
        sum(v < 12 for v in results) / samples
    )

print("MOBILE COMPUTING - ASSIGNMENT 1")
print("FREQUENCY REUSE AND SIR SIMULATION")
print("-" * 65)
print("Path loss | Interferers | Cluster | Channels | Capacity | Sites")
print("-" * 65)

for n in (2.5, 3, 4):
    for k, sectors in ((6, 1), (2, 3), (1, 6)):
        N = minimum_N(n, k)
        c = S // (N * sectors)
        a = sectors * capacity(c)
        sites = max(5, math.ceil(2500 / a))

        mean_sir, fifth_sir, outage = sample_sir(N, n, k)

        print(
            f"{n:8} | {k:11} | {N:7} | {c:8} | "
            f"{a:8.2f} | {sites:5}"
        )
        print(
            f"   Mean SIR: {mean_sir:.2f} dB | "
            f"5th percentile: {fifth_sir:.2f} dB | "
            f"Outage: {outage*100:.2f}%"
        )

print("-" * 65)
print("Validation check:",
      round(10 * math.log10((3*7)**2 / 6), 1), "dB")
print("Simulation completed successfully.")
