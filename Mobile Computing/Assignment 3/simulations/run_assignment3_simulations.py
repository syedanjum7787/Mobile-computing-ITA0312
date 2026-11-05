#!/usr/bin/env python3
"""Mobile Computing Assignment 3: reproducible Python demonstrations.
All outputs are PYTHON ILLUSTRATIVE, not NS-2/NS-3 experiments.
Source: Mobile_Computing_Assignment_3_CO3.pdf (36-page report).
"""
from pathlib import Path
from collections import deque
import csv
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA, FIG = ROOT / 'data', ROOT / 'figures'
DATA.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
SEED = 42

def write(name, header, rows):
    with (DATA / name).open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

def plot_save(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=170, bbox_inches='tight')
    plt.close()

def graph(edges):
    g = {}
    for a, b in edges:
        g.setdefault(a, set()).add(b)
        g.setdefault(b, set()).add(a)
    return g

def shortest(g, src, dst):
    todo = deque([[src]])
    visited = {src}
    while todo:
        path = todo.popleft()
        if path[-1] == dst:
            return path
        for n in sorted(g.get(path[-1], ())):
            if n not in visited:
                visited.add(n)
                todo.append(path + [n])
    return None

# Q1: Illustration of handoff in a walking campus scenario.
# Assumptions for *this* model: APs at 0 and 100 m, path 0..100 m,
# received signal model -40 - 20log10(1 + separation/m), threshold -78 dBm,
# hysteresis 3 dB and a 0.3 second interruption.
def q1():
    loc = np.linspace(0, 100, 201)
    r1 = -40 - 20 * np.log10(1 + np.abs(loc) / 1)
    r2 = -40 - 20 * np.log10(1 + np.abs(100 - loc) / 1)
    current, start, rows = 'AP1', None, []
    for i, x in enumerate(loc):
        t = i * .1
        s1, s2 = float(r1[i]), float(r2[i])
        if current == 'AP1' and s1 < -78 and s2 >= -78 and s2 >= s1 + 3:
            current, start = 'AP2', t
        interruption = start is not None and t < start + .3 - 1e-9
        rows.append((round(t, 2), round(float(x), 2), round(s1, 3), round(s2, 3), current,
                     int(not interruption), 'HANDOFF' if interruption else 'CONNECTED'))
    write('q1_handoff_illustrative.csv', ['time_s','position_m','ap1_dbm','ap2_dbm','serving_ap','connected','state'], rows)
    plt.figure(figsize=(8, 4))
    plt.plot(loc, r1, label='AP1 RSSI (assumed)')
    plt.plot(loc, r2, label='AP2 RSSI (assumed)')
    plt.axhline(-78, linestyle=':', label='Assumed threshold')
    if start is not None:
        plt.axvline(start * 5, linestyle='--', label='Handoff trigger')
    plt.xlabel('Distance along path (m)'); plt.ylabel('Received power (dBm)')
    plt.title('Q1 – Illustrative campus handoff'); plt.legend()
    plot_save('q1_handoff.png')

# Q2: Event trace; no real IP stack or network socket employed.
def q2():
    events = [
        (0,'MN','FA1','agent advertisement','discovery'),
        (1,'MN','FA1','registration request (CoA1)','registration'),
        (2,'FA1','HA','forward registration request','registration'),
        (3,'HA','FA1','registration accepted; HoA -> CoA1','binding update'),
        (4,'CN','HA','packet to HoA','delivery'),
        (5,'HA','FA1','IP-in-IP encapsulation to CoA1','tunnel'),
        (6,'FA1','MN','decapsulate and deliver','delivery'),
        (7,'MN','FA2','agent advertisement / movement','discovery'),
        (8,'MN','FA2','registration request (CoA2)','registration'),
        (9,'FA2','HA','forward registration request','registration'),
        (10,'HA','FA2','registration accepted; HoA -> CoA2','binding update'),
        (11,'HA','FA2','tunnel subsequent packet to CoA2','tunnel'),
        (12,'FA2','MN','decapsulate and deliver','delivery'),
    ]
    write('q2_mobile_ip_event_trace.csv', ['step','source','destination','event','phase'], events)
    write('q2_mobile_ip_binding.csv', ['step','home_address','care_of_address','registration_status'],
          [(3,'HoA','CoA1','REGISTERED'), (10,'HoA','CoA2','REGISTERED')])
    plt.figure(figsize=(9, 4))
    phases = ['discovery','registration','binding update','delivery','tunnel']
    plt.scatter([x[0] for x in events], [phases.index(x[4]) for x in events], s=65)
    plt.yticks(range(len(phases)), phases); plt.xlabel('Event step')
    plt.title('Q2 – Conceptual Mobile IP registration and tunnelling trace')
    plot_save('q2_mobile_ip_events.png')

# Q3: source-derived, qualitative table (not measured).
def q3():
    rows = [
        ('Infrastructure','Base stations + core','None required'),
        ('Control','Centralised','Distributed'),
        ('Topology','Cellular coverage','Dynamic multi-hop'),
        ('Routing','Core-network routed','Nodes route via DSDV / AODV'),
        ('Mobility','Handoff','Route repair/discovery'),
        ('Disaster resilience','Depends on infrastructure','Locally self-forming'),
        ('Security','Central authentication','Distributed security needed'),
        ('Power','Single-hop MS transmission','Nodes relay for others'),
    ]
    write('q3_architecture_comparison.csv', ['parameter','cellular','ad_hoc'], rows)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set(xlim=(0, 10), ylim=(0, 4)); ax.axis('off')
    boxes = [('BS1',(1.5,2.6)),('BS2',(4,2.6)),('Core',(2.75,3.5)),
             ('R1',(6.5,3)),('R2',(8.5,3)),('R3',(7.5,1.5))]
    for a,b in [('BS1','Core'),('BS2','Core'),('R1','R2'),('R2','R3'),('R1','R3')]:
        p=dict(boxes); x,y=p[a]; x2,y2=p[b]; ax.plot([x,x2],[y,y2],color='gray',lw=2,zorder=1)
    for label,(x,y) in boxes:
        ax.text(x,y,label,ha='center',va='center',bbox=dict(boxstyle='round',fc='white',ec='black'),zorder=2)
    ax.text(2.75,.55,'Cellular (infrastructure)',ha='center')
    ax.text(7.5,.55,'Ad hoc (multi-hop)',ha='center')
    ax.set_title('Q3 – Qualitative architectural comparison')
    plot_save('q3_architectures.png')

# Q4: telemedicine communication path and requirement table from PDF.
def q4():
    rows = [
        ('Ambulance router','Mobile node; patient monitor/camera behind router'),
        ('Visited cellular gateway','Foreign Agent / care-of address'),
        ('Hospital gateway','Home Agent / binding table'),
        ('Hospital server','Correspondent Node / physician endpoint'),
        ('Delay','About 150 ms one-way interactive media DESIGN TARGET, not verified'),
        ('Security','Authenticated registration, IPsec/VPN, encryption'),
        ('Resilience','Backup HA and second-operator fallback'),
    ]
    write('q4_ambulance_design_requirements.csv',['component_or_requirement','role_or_design_target'],rows)
    steps = [('Ambulance MN','FA'),('FA','HA'),('HA','FA'),('FA','Ambulance MN'),
             ('Hospital CN','HA'),('HA','FA'),('FA','Ambulance MN')]
    write('q4_ambulance_communication_flow.csv',['step','from','to'],[(i+1,a,b) for i,(a,b) in enumerate(steps)])
    fig, ax=plt.subplots(figsize=(10,3.5)); ax.axis('off')
    labels=['Ambulance MN','Visited FA','Hospital HA','Hospital CN']
    xx=[.12,.36,.62,.87]
    for x,label in zip(xx,labels):
        ax.text(x,.55,label,ha='center',va='center',transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=.65',ec='black',fc='white'))
    for a,b in zip(xx[:-1],xx[1:]):
        ax.annotate('',xy=(b-.065,.55),xytext=(a+.075,.55),xycoords='axes fraction',
                    arrowprops=dict(arrowstyle='->',lw=2))
    ax.text(.5,.2,'Conceptual Mobile IP registration and hospital data path',ha='center',transform=ax.transAxes)
    ax.set_title('Q4 – Ambulance telemedicine architecture')
    plot_save('q4_ambulance_architecture.png')

# Q5: exact PDF topology and specified D-100 -> D-101 -> D-102 transitions.
def q5():
    before=graph([('A','B'),('B','C'),('B','E'),('C','D'),('E','D')])
    after=graph([('A','B'),('B','C'),('B','E'),('E','D')])
    rows=[]
    for stage,g,seq in [('before',before,100),('converged',after,102)]:
        for src in ['A','B']:
            for dest in ['A','B','C','D','E']:
                if src==dest: continue
                path=shortest(g,src,dest)
                rows.append((stage,src,dest,path[1] if path else '',len(path)-1 if path else 'INF',
                             seq if dest=='D' else 100,'-'.join(path) if path else 'unreachable'))
    write('q5_dsdv_routing_tables.csv', ['stage','source','destination','next_hop','hop_count','dest_sequence','path'], rows)
    transition=[('before','B','D','C',2,100,'valid'),
                ('broken-link advertisement','B','D','C','INF',101,'invalid'),
                ('after higher-seq update','B','D','E',2,102,'valid')]
    write('q5_dsdv_link_failure_events.csv', ['stage','source','destination','next_hop','metric','destination_sequence','status'],transition)
    fig,ax=plt.subplots(figsize=(8,4)); ax.axis('off')
    positions={'A':(0,1),'B':(1,1),'C':(2,2),'E':(2,0),'D':(3,1)}
    for a,b in [('A','B'),('B','C'),('B','E'),('E','D')]:
        x,y=positions[a]; x2,y2=positions[b]; ax.plot([x,x2],[y,y2],color='black',lw=1.8)
    x,y=positions['C'];x2,y2=positions['D']
    ax.plot([x,x2],[y,y2],color='red',lw=2,ls='--',label='Failed C–D')
    for n,(x,y) in positions.items(): ax.text(x,y,n,ha='center',va='center',bbox=dict(boxstyle='circle,pad=.5',fc='white',ec='black'))
    ax.set_xlim(-.5,3.6);ax.set_ylim(-.5,2.6);ax.legend(loc='lower left')
    ax.set_title('Q5 – PDF topology: DSDV recovery via B–E–D')
    plot_save('q5_dsdv_topology.png')

# Q6: simplified graph-based route discovery and repair; not a full AODV protocol stack.
def q6():
    before=graph([('A','B'),('B','C'),('C','D'),('B','E'),('E','D')])
    after=graph([('A','B'),('B','C'),('B','E'),('E','D')])
    path1=shortest(before,'A','D')
    path2=shortest(after,'A','D')
    logs=[(1,'RREQ','A','D','broadcast conceptual discovery'),
          (2,'RREP','D','A','-'.join(path1)),
          (3,'LINK_BREAK','C','D','C–D link down'),
          (4,'RERR','C','A','invalidate route using C–D'),
          (5,'RREQ','A','D','new conceptual discovery'),
          (6,'RREP','D','A','-'.join(path2))]
    write('q6_aodv_event_trace.csv',['step','event','source','destination','details'],logs)
    write('q6_aodv_paths.csv',['stage','source','destination','route','hops'],
          [('before_failure','A','D','-'.join(path1),len(path1)-1),
           ('after_repair','A','D','-'.join(path2),len(path2)-1)])
    fig,ax=plt.subplots(figsize=(9,3));ax.axis('off')
    ax.text(.5,.72,'A → B → C → D   (original discovered route)',ha='center',fontsize=13,transform=ax.transAxes)
    ax.text(.5,.45,'C–D fails → RERR → new RREQ / RREP',ha='center',fontsize=12,transform=ax.transAxes)
    ax.text(.5,.18,'A → B → E → D   (repaired route)',ha='center',fontsize=13,transform=ax.transAxes)
    ax.set_title('Q6 – Illustrative AODV discovery and link-repair events')
    plot_save('q6_aodv_repair.png')

# Q7: EXACT source-report illustrative parameters: 10 ms/hop, 100 ms
# backhaul, 5 ms tunnel; 4 hops gives 145 ms steady and +80 ms discovery.
# All other hop counts are formula-based derivations, not measured data.
def q7():
    rows=[]
    for k in range(1,9):
        base=100+5+10*k
        discover=2*k*10
        rows.append((k,10,100,5,base,discover,base+discover,'ANALYTICAL_ILLUSTRATION'))
    write('q7_analytical_delay.csv', ['hops','assumed_hop_delay_ms','assumed_backhaul_ms',
          'assumed_tunnel_ms','steady_one_way_delay_ms','extra_first_packet_discovery_ms',
          'illustrative_first_packet_delay_ms','evidence_type'],rows)
    write('q7_ns3_experimental_results_TEMPLATE.csv',
          ['speed_scenario','protocol','mean_delay_ms','packet_delivery_ratio_percent',
           'routing_control_packets_per_delivered_data_packet','source','status'],
          [(speed,proto,'','','','NS-2/NS-3 experiment required','NOT_MEASURED')
           for speed in ['Low','Medium','High'] for proto in ['AODV','DSDV']])
    write('q7_ns3_experiment_plan.csv', ['parameter','planned_setting','status'],[
        ('scenario','flood disaster emergency ad hoc network','PLANNED'),
        ('nodes','20–50 rescue nodes','PLANNED'),
        ('gateways','2 gateway nodes','PLANNED'),
        ('mobility','random waypoint, low/medium/high speed; exact speeds not supplied','TO_SPECIFY'),
        ('flows','CBR voice-like flows to command centre; traffic parameters not supplied','TO_SPECIFY'),
        ('routing','AODV vs DSDV (DSDV support to be independently verified in chosen simulator)','PLANNED'),
        ('Mobile IP','HA/FA gateway handling requires an actual compatible implementation','TO_IMPLEMENT'),
        ('failures','inject link and gateway failures; event times not supplied','TO_SPECIFY'),
        ('metrics','mean end-to-end delay; PDR; control packet overhead','PLANNED'),
        ('experimental results','none provided in original PDF','NOT_MEASURED'),
    ])
    fig,ax=plt.subplots(figsize=(8,4));ax.plot([r[0] for r in rows],[r[4] for r in rows],marker='o',label='Steady analytical delay')
    ax.plot([r[0] for r in rows],[r[6] for r in rows],marker='s',label='First packet incl. discovery')
    ax.set(xlabel='Hop count to gateway',ylabel='Assumed end-to-end delay (ms)',
           title='Q7 – Formula-based illustration, NOT NS-2/NS-3 measurements')
    ax.legend();plot_save('q7_analytical_delay.png')

if __name__=='__main__':
    for num,fn in enumerate((q1,q2,q3,q4,q5,q6,q7),1):
        fn()
        print(f'Q{num}: generated')
    print(f'All generated files saved to {ROOT}')
