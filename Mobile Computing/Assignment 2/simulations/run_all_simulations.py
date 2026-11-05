"""
Mobile Computing Assignment 2 (CO2)
Reproducible simulations for Q1-Q7.
Run from the Assignment-2 folder:
    python3 simulations/run_all_simulations.py
Outputs:
    data/q1_results.csv ... data/q7_validation.csv
    figures/q1_simulation.png ... figures/q7_validation.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import erfc

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "figures"
DATA.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
rng = np.random.default_rng(42)

def save_plot(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=200, bbox_inches="tight")
    plt.close()

# Q1: Offline synchronization; two-state link availability and buffering.
def q1():
    rows = []
    for availability in (0.9, 0.6, 0.3):
        n, horizon = 1000, 1000
        # Generate event times uniformly across the shift and an availability
        # process with a fixed mean outage duration of four minutes.
        times = np.sort(rng.uniform(0, horizon, n))
        state = rng.random() < availability
        link = np.zeros(horizon + 1, dtype=bool)
        t = 0
        while t <= horizon:
            mean = (availability / (1-availability))*4 if state else 4
            duration = max(1, int(rng.exponential(mean)))
            link[t:min(horizon+1, t+duration)] = state
            t += duration
            state = not state
        delivered = link[np.minimum(times.astype(int), horizon)]
        # Approximate store-and-forward wait as time to next available minute.
        delays = []
        for ti in times:
            idx = min(int(ti), horizon)
            future = np.flatnonzero(link[idx:])
            delays.append(float(future[0]) if len(future) else float(horizon-idx))
        for strategy in ("Immediate Send", "Store-and-Forward"):
            if strategy == "Immediate Send":
                count = int(delivered.sum())
                delay = np.zeros(count)
                lost = n-count
            else:
                count, lost = n, 0
                delay = np.asarray(delays)
            rows.append({"availability": availability, "strategy": strategy,
                         "generated": n, "delivered": count, "lost": lost,
                         "success_percent": 100*count/n,
                         "mean_delay_min": float(delay.mean()) if len(delay) else 0,
                         "max_delay_min": float(delay.max()) if len(delay) else 0})
    df = pd.DataFrame(rows)
    df.to_csv(DATA/"q1_results.csv", index=False)
    pivot = df.pivot(index="availability", columns="strategy", values="success_percent")
    pivot.plot(kind="bar", ylabel="Delivery success (%)", xlabel="Link availability",
               title="Q1: Immediate Send vs Store-and-Forward", ylim=(0, 105))
    save_plot("q1_simulation.png")

# Q2: Compare dedicated FDMA/TDMA channel pools under four traffic patterns.
def q2():
    rows = []
    nreq = 5000
    for scenario in ("Voice", "Bursty Data", "70/30 Mix", "Flash Telemetry"):
        # Fixed seed-derived request durations and arrivals; Erlang-loss model.
        if scenario == "Flash Telemetry":
            arrivals = np.repeat(np.arange(50)*60.0, 100)
            durations = rng.uniform(5, 15, len(arrivals))
        else:
            arrivals = np.sort(rng.uniform(0, 3600, nreq))
            durations = rng.exponential(120, nreq)
        for scheme, capacity in (("FDMA", 74), ("TDMA", 78)):
            ends = []
            blocked = 0
            busy_area = 0.0
            last_t = 0.0
            for a, d in zip(arrivals, durations):
                # Remove calls completed before this arrival.
                ends = [e for e in ends if e > a]
                if len(ends) < capacity:
                    ends.append(a+d)
                else:
                    blocked += 1
            block_pct = 100*blocked/len(arrivals)
            duty = {"Voice":1.0, "Bursty Data":0.1, "70/30 Mix":0.73,
                    "Flash Telemetry":1.0}[scenario]
            rows.append({"scenario":scenario, "scheme":scheme, "channels":capacity,
                         "requests":len(arrivals), "blocked":blocked,
                         "blocking_percent":block_pct, "nominal_utilization_percent":
                         100*(1-block_pct/100)*min(1, len(arrivals)*durations.mean()/(3600*capacity)),
                         "effective_utilization_percent":100*(1-block_pct/100)*
                         min(1, len(arrivals)*durations.mean()/(3600*capacity))*duty,
                         "wasted_capacity_percent":100*(1-duty) if scenario!="Flash Telemetry" else 0})
    df = pd.DataFrame(rows)
    df.to_csv(DATA/"q2_results.csv", index=False)
    p = df.pivot(index="scenario", columns="scheme", values="blocking_percent")
    p.plot(kind="bar", ylabel="Blocked requests (%)", title="Q2: FDMA vs TDMA blocking")
    save_plot("q2_simulation.png")

# Q3: DS-CDMA BPSK BER versus user count for Walsh and random spreading.
def q3():
    rows=[]
    Nbits=5000
    ebn0=10**(10/10)
    for N in (8,16,64):
        # Hadamard/Walsh matrix.
        H=np.array([[1]], dtype=float)
        while H.shape[0] < N:
            H=np.block([[H,H],[H,-H]])
        for K in sorted(set([2, max(2,N//2), N, int(1.25*N)])):
            bits=rng.choice([-1,1], size=(K,Nbits))
            noise=rng.normal(0, np.sqrt(1/(2*ebn0)), size=(Nbits,N))
            # Synchronous orthogonal Walsh: project received chips onto user's code.
            for condition in ("Walsh Sync","Random Codes","Near-Far 10x"):
                codes=H[np.arange(K) % N].copy() if condition!="Random Codes" else rng.choice([-1,1],(K,N))
                if condition=="Near-Far 10x" and K>1:
                    powers=np.ones(K); powers[1]=10
                else:
                    powers=np.ones(K)
                chips=bits[:,:,None]*codes[:,None,:]*np.sqrt(powers)[:,None,None]
                # Sum all users over chip axis; first user's matched filter.
                rx=chips.sum(axis=0) + noise
                decision=(rx @ codes[0]) / N
                errors=np.count_nonzero(np.sign(decision)!=bits[0])
                rows.append({"code_length":N,"processing_gain_db":10*np.log10(N),
                             "users":K,"condition":condition,"bit_errors":int(errors),
                             "bits_tested":Nbits,"ber":errors/Nbits})
    df=pd.DataFrame(rows); df.to_csv(DATA/"q3_results.csv",index=False)
    for condition,g in df.groupby("condition"):
        for N,h in g.groupby("code_length"):
            plt.plot(h["users"],h["ber"],marker="o",label=f"{condition}, N={N}")
    plt.yscale("log"); plt.xlabel("Number of users"); plt.ylabel("BER")
    plt.title("Q3: DS-CDMA BER"); plt.legend(fontsize=7)
    save_plot("q3_simulation.png")

# Q4: OFDMA orthogonality and three allocation strategies over fading channels.
def q4():
    K,M,realizations=8,64,1000
    rows=[]
    for r in range(realizations):
        distances=np.linspace(100,800,K)
        path=(100/distances)**3.5
        # Frequency-selective Rayleigh channel with independent subcarrier fading.
        fading=(rng.normal(size=(K,M))+1j*rng.normal(size=(K,M)))/np.sqrt(2)
        gain=path[:,None]*np.abs(fading)**2
        rate=np.log2(1+gain*160)
        allocations={
            "Contiguous Blocks":np.repeat(np.arange(K),M//K),
            "Interleaved Round-Robin":np.arange(M)%K,
            "Best-Subcarrier":np.argmax(gain,axis=0)
        }
        for scheme, owners in allocations.items():
            per=np.array([rate[u, owners==u].sum() for u in range(K)])
            counts=np.bincount(owners,minlength=K)
            jain=(per.sum()**2)/(K*np.sum(per**2)) if np.sum(per**2)>0 else 0
            rows.append({"realization":r,"scheme":scheme,"sum_rate_bps_hz":per.sum(),
                         "jain_fairness":jain,"min_user_rate":per.min(),
                         "max_user_rate":per.max(),"starved_users_percent":100*np.mean(per==0),
                         "allocated_subcarriers_min":counts.min(),
                         "allocated_subcarriers_max":counts.max()})
    df=pd.DataFrame(rows); df.to_csv(DATA/"q4_results.csv",index=False)
    summary=df.groupby("scheme")[["sum_rate_bps_hz","jain_fairness","min_user_rate",
                                  "max_user_rate","starved_users_percent"]].mean()
    summary.to_csv(DATA/"q4_validation.csv")
    summary[["sum_rate_bps_hz","jain_fairness"]].plot(kind="bar",title="Q4: Throughput and fairness")
    save_plot("q4_simulation.png")

# Q5: ULA array factor and SIR versus angular separation.
def q5():
    theta=np.linspace(-90,90,1801)
    rows=[]
    plt.figure()
    for N in (2,4,8,16):
        psi=np.pi*(np.sin(np.deg2rad(theta))-np.sin(0))
        den=N*np.sin(psi/2)
        af=np.ones_like(theta)
        mask=np.abs(den)>1e-12
        af[mask]=np.abs(np.sin(N*psi[mask]/2)/den[mask])
        af_db=20*np.log10(np.maximum(af,1e-6))
        plt.plot(theta,af_db,label=f"N={N}")
        for sep in (5,10,15,30,45,60):
            p=np.pi*(np.sin(np.deg2rad(sep))-0)
            val=abs(np.sin(N*p/2)/(N*np.sin(p/2))) if abs(np.sin(p/2))>1e-12 else 1
            rows.append({"elements":N,"separation_deg":sep,
                         "sir_db":-20*np.log10(max(val,1e-6))})
    plt.xlabel("Angle (degrees)"); plt.ylabel("Normalized array factor (dB)")
    plt.title("Q5: ULA radiation patterns"); plt.ylim(-60,5); plt.legend()
    save_plot("q5_simulation.png")
    df=pd.DataFrame(rows); df.to_csv(DATA/"q5_results.csv",index=False)
    df.pivot(index="separation_deg",columns="elements",values="sir_db").to_csv(DATA/"q5_validation.csv")

# Q6: Weighted decision matrix using scenario weights from the report.
def q6():
    criteria=["Spectral Efficiency","Interference","Simplicity","Mobility","Scalability"]
    schemes=["SDMA","TDMA","FDMA","CDMA","OFDMA"]
    # Transparent illustrative 1-10 ratings, applied to stated report weights.
    ratings=pd.DataFrame([[7,6,5,7,7],[5,8,8,7,6],[5,6,9,6,7],
                          [8,9,4,7,7],[10,7,5,8,9]],
                         index=schemes,columns=criteria)
    weights={"Cellular Voice":[.25,.15,.20,.25,.15],
             "Tactical Link":[.10,.40,.15,.20,.15],
             "Stadium Downlink":[.35,.15,.10,.15,.25],
             "5000 Sensors":[.10,.15,.35,.15,.25]}
    rows=[]
    for scenario,w in weights.items():
        scores=ratings.to_numpy()@np.array(w)
        for s,score in zip(schemes,scores):
            rows.append({"scenario":scenario,"scheme":s,"weighted_score":score})
    df=pd.DataFrame(rows); df.to_csv(DATA/"q6_results.csv",index=False)
    df.pivot(index="scenario",columns="scheme",values="weighted_score").to_csv(DATA/"q6_validation.csv")
    df.pivot(index="scenario",columns="scheme",values="weighted_score").plot(kind="bar")
    plt.ylabel("Weighted score (1-10)"); plt.title("Q6: Multiple-access selection matrix")
    save_plot("q6_simulation.png")

# Q7: Compare greedy best-subcarrier with quota-constrained proportional fairness.
def q7():
    K,M,R=8,64,1000
    distances=np.linspace(100,800,K)
    path=(100/distances)**3.5
    accum=[]
    for r in range(R):
        fading=(rng.normal(size=(K,M))+1j*rng.normal(size=(K,M)))/np.sqrt(2)
        gain=path[:,None]*np.abs(fading)**2
        rate=np.log2(1+gain*160)
        greedy=np.argmax(gain,axis=0)
        # Quota phase: allocate 5 best available subcarriers to farthest users first.
        owners=np.full(M,-1,dtype=int)
        for u in range(K-1,-1,-1):
            available=np.flatnonzero(owners==-1)
            best=available[np.argsort(gain[u,available])[-5:]]
            owners[best]=u
        # Proportional-fair phase on remaining subcarriers.
        base=np.array([rate[u,owners==u].sum() for u in range(K)])
        for m in np.flatnonzero(owners==-1):
            metric=rate[:,m]/(base+0.5)
            u=int(np.argmax(metric)); owners[m]=u; base[u]+=rate[u,m]
        for name,allocation in (("Best-Subcarrier",greedy),("QC-PF",owners)):
            per=np.array([rate[u,allocation==u].sum() for u in range(K)])
            counts=np.bincount(allocation,minlength=K)
            accum.append({"realization":r,"scheme":name,"sum_throughput":per.sum(),
                          "jain_fairness":per.sum()**2/(K*np.sum(per**2)),
                          "min_user_rate":per.min(),"p5_user_rate":np.percentile(per,5),
                          "starved_users_percent":100*np.mean(per==0),
                          "min_subcarriers":counts.min(),"max_subcarriers":counts.max()})
    df=pd.DataFrame(accum); df.to_csv(DATA/"q7_results.csv",index=False)
    summary=df.groupby("scheme").mean(numeric_only=True)
    summary.to_csv(DATA/"q7_validation.csv")
    summary[["sum_throughput","jain_fairness","min_user_rate","starved_users_percent"]].plot(kind="bar")
    plt.title("Q7: Greedy vs quota-constrained proportional fairness")
    save_plot("q7_validation.png")

if __name__=="__main__":
    q1(); q2(); q3(); q4(); q5(); q6(); q7()
    print("Assignment 2 simulations completed.")
    print(f"CSV files saved in: {DATA}")
    print(f"PNG graphs saved in: {FIG}")
    print("Generated files:")
    for p in sorted(DATA.glob("*.csv")) + sorted(FIG.glob("*.png")):
        print(p.relative_to(ROOT))
