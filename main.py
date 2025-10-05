from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Callable, Any
import heapq
import itertools
import numpy as np
import math
import time

# Config

@dataclass
class SimConfig:
    seed: int = 123
    dt_ms: int = 100                           # simulation tick
    sim_time_ms: int = 2 * 60 * 1000           # total runtime (example: 2 minutes)
    # Nodes / accounts
    n_nodes: int = 8
    init_balance: float = 100.0
    hashrate_base_hs: float = 1e9              # 1 GH/s baseline
    hashrate_jitter_sigma: float = 0.4         # log-normal jitter (exp(N(0,sigma)))
    # Latency (Gaussian, clipped)
    node_latency_mean_mu: float = 60.0         # ms, per-node mean drawn from N(mu, sigma)
    node_latency_mean_sigma: float = 20.0
    node_latency_std_scale: float = 0.25       # std ~= scale * mean (>= min below)
    node_latency_std_min: float = 5.0
    # Edges / link layer
    default_bandwidth_kbps: float = 5000.0     # serialize cost = size_kB / (bw_kBps)
    link_drop_prob: float = 0.01               # Bernoulli
    # Processing (Gaussian, clipped)
    node_proc_mean_ms: float = 5.0
    node_proc_std_ms: float = 2.0
    # Transaction generation (Bernoulli per-tick)
    tx_create_p: float = 0.10
    tx_amount_mean: float = 1.0
    tx_fee_mean: float = 0.01
    tx_size_kB_mean: float = 0.5
    # Blocks / PoW (optional for routing, but included to track "blocks in flight")
    target_block_time_ms: int = 25_000
    block_size_txs: int = 800
    coinbase_reward: float = 6.25

# =========================
# Entities
# =========================

@dataclass
class Transaction:
    tx_id: int
    src: int
    dst: int
    value: float
    fee: float
    size_kB: float
    created_ms: int
    deadline_ms: Optional[int] = None          # for QoS experiments later
    path_hint: Optional[List[int]] = None      # optional fixed path (for baselines)

@dataclass
class Block:
    block_id: int
    miner: int
    parent_id: Optional[int]
    timestamp_ms: int
    txs: List[Transaction] = field(default_factory=list)

@dataclass
class Node:
    node_id: int
    balance: float
    hashrate_hs: float
    latency_mean_ms: float
    latency_std_ms: float
    proc_mean_ms: float
    proc_std_ms: float
    neighbors: List[int] = field(default_factory=list)
    # Mempool & chain view (simplified)
    mempool: Dict[int, Transaction] = field(default_factory=dict)
    seen_blocks: set[int] = field(default_factory=set)
    blocks: Dict[int, Block] = field(default_factory=dict)
    heights: Dict[int, int] = field(default_factory=dict)
    head_id: Optional[int] = None

@dataclass
class Edge:
    u: int
    v: int
    bandwidth_kBps: float
    # Additional per-edge attributes can go here (policy score, trust, etc.)

@dataclass
class PathIntent:
    """A routing request: move a transaction from src to dst minimizing some cost."""
    tx_id: int
    src: int
    dst: int
    created_ms: int

# =========================
# Events & Queue
# =========================

@dataclass(order=True)
class Event:
    deliver_ms: int
    seq: int
    kind: str = field(compare=False)
    payload: Dict[str, Any] = field(compare=False)

class EventQueue:
    def __init__(self):
        self._pq: List[Event] = []
        self._seq = itertools.count()

    def push(self, when_ms: int, kind: str, payload: Dict[str, Any]):
        heapq.heappush(self._pq, Event(when_ms, next(self._seq), kind, payload))

    def pop_ready(self, now_ms: int) -> List[Event]:
        out = []
        while self._pq and self._pq[0].deliver_ms <= now_ms:
            out.append(heapq.heappop(self._pq))
        return out

    def count_inflight_blocks(self) -> int:
        return sum(1 for ev in self._pq if ev.kind == "deliver_block")

    def count_inflight_txs(self) -> int:
        return sum(1 for ev in self._pq if ev.kind == "deliver_tx")

# =========================
# Random sources (Gauss/Bernoulli)
# =========================

class Rand:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def gauss_clip(self, mu: float, sigma: float, min_val: float = 0.0) -> float:
        x = self.rng.normal(mu, sigma)
        return float(max(min_val, x))

    def bernoulli(self, p: float) -> bool:
        return bool(self.rng.binomial(1, p))

    def lognormal_like(self, sigma: float) -> float:
        # exp(N(0,sigma)) ~ log-normal with median=1
        return float(np.exp(self.rng.normal(0.0, sigma)))

    def choice(self, seq: List[int], p: Optional[np.ndarray] = None) -> int:
        return int(self.rng.choice(seq, p=p))

# =========================
# State & Metrics
# =========================

@dataclass
class NetworkState:
    """Snapshot suitable for RL observation later (kept simple for now)."""
    time_ms: int
    balances: Dict[int, float]
    hashrates: Dict[int, float]
    latency_means: Dict[int, float]
    mempool_sizes: Dict[int, int]
    inflight_blocks: int
    inflight_txs: int

@dataclass
class Metrics:
    blocks_mined: int = 0
    blocks_stale: int = 0
    tx_injected: int = 0
    tx_delivered: int = 0
    tx_dropped: int = 0
    tx_avg_end_to_end_ms: float = 0.0   # EWMA
    orphan_rate: float = 0.0            # derived: stale / mined

# =========================
# Router interface (NO optimization now)
# =========================

class BaseRouter:
    """Strategy that decides which neighbor to forward a TX to.
    DO NOT implement optimization here; this is an interface and placeholder."""
    def route_next_hop(self, node_id: int, tx: Transaction, neighbors: List[int], state: "Sim") -> Optional[int]:
        raise NotImplementedError

class PlaceholderRouter(BaseRouter):
    """A trivial router you can replace later.
    - If the destination is an immediate neighbor, use it.
    - Otherwise, pick a neighbor uniformly at random.
    """
    def route_next_hop(self, node_id: int, tx: Transaction, neighbors: List[int], state: "Sim") -> Optional[int]:
        if tx.dst in neighbors:
            return tx.dst
        if not neighbors:
            return None
        return state.rand.choice(neighbors)

# =========================
# Simulator
# =========================

class Sim:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.rand = Rand(cfg.seed)
        self.time_ms = 0
        self.q = EventQueue()
        self.metrics = Metrics()
        self.router: BaseRouter = PlaceholderRouter()  # swap later

        # Build nodes with per-node latency/processing drawn from Gaussians
        self.nodes: Dict[int, Node] = {}
        for i in range(cfg.n_nodes):
            lat_mean = self.rand.gauss_clip(cfg.node_latency_mean_mu, cfg.node_latency_mean_sigma, min_val=1.0)
            lat_std = max(cfg.node_latency_std_min, cfg.node_latency_std_scale * lat_mean)
            proc_mean = cfg.node_proc_mean_ms
            proc_std = max(0.1, cfg.node_proc_std_ms)
            hashrate = cfg.hashrate_base_hs * self.rand.lognormal_like(cfg.hashrate_jitter_sigma)
            self.nodes[i] = Node(
                node_id=i,
                balance=cfg.init_balance,
                hashrate_hs=hashrate,
                latency_mean_ms=lat_mean,
                latency_std_ms=lat_std,
                proc_mean_ms=proc_mean,
                proc_std_ms=proc_std,
            )

        # Fully connect (undirected edges)
        self.edges: Dict[Tuple[int, int], Edge] = {}
        for i in self.nodes:
            for j in self.nodes:
                if i == j: continue
                self.nodes[i].neighbors.append(j)
                self.edges[(i, j)] = Edge(u=i, v=j, bandwidth_kBps=cfg.default_bandwidth_kbps)

        # Genesis block on each node
        self.block_seq = itertools.count(start=1)
        self.tx_seq = itertools.count(start=1)
        genesis = Block(block_id=0, miner=-1, parent_id=None, timestamp_ms=0)
        for n in self.nodes.values():
            n.blocks[0] = genesis
            n.heights[0] = 0
            n.head_id = 0
            n.seen_blocks.add(0)

    # ---------- Distributions ----------

    def sample_link_latency_ms(self, u: int, v: int) -> int:
        a, b = self.nodes[u], self.nodes[v]
        mu = 0.5 * (a.latency_mean_ms + b.latency_mean_ms)
        sigma = 0.5 * (a.latency_std_ms + b.latency_std_ms)
        return int(self.rand.gauss_clip(mu, sigma, min_val=1.0))

    def sample_proc_ms(self, node_id: int) -> int:
        n = self.nodes[node_id]
        return int(self.rand.gauss_clip(n.proc_mean_ms, n.proc_std_ms, min_val=0.0))

    def bern_drop(self) -> bool:
        return self.rand.bernoulli(self.cfg.link_drop_prob)

    # ---------- Transactions ----------

    def maybe_inject_tx(self):
        if not self.rand.bernoulli(self.cfg.tx_create_p):
            return
        ids = list(self.nodes.keys())
        # weight source by balance (richer emits more)
        bal = np.array([max(0.0, self.nodes[i].balance) for i in ids], dtype=float)
        p = (bal + 1.0) / (bal.sum() + len(ids))
        src = self.rand.choice(ids, p=p)
        dst = self.rand.choice([i for i in ids if i != src])
        value = max(0.0001, float(self.rand.gauss_clip(self.cfg.tx_amount_mean, 0.25 * self.cfg.tx_amount_mean, 0.0001)))
        fee   = max(0.00001, float(self.rand.gauss_clip(self.cfg.tx_fee_mean,   0.20 * self.cfg.tx_fee_mean,   0.00001)))
        size_kB = max(0.01, float(self.rand.gauss_clip(self.cfg.tx_size_kB_mean, 0.15 * self.cfg.tx_size_kB_mean, 0.01)))
        tx = Transaction(tx_id=next(self.tx_seq), src=src, dst=dst, value=value, fee=fee, size_kB=size_kB, created_ms=self.time_ms)
        self.metrics.tx_injected += 1
        # Add to src mempool and trigger local forwarding
        self.nodes[src].mempool[tx.tx_id] = tx
        self.schedule_tx_forward(src, tx)

    def schedule_tx_forward(self, node_id: int, tx: Transaction):
        """Ask router for the next hop and schedule delivery with latency + serialization + processing."""
        node = self.nodes[node_id]
        hop = self.router.route_next_hop(node_id, tx, node.neighbors, self)
        if hop is None:
            return  # nowhere to go
        # link drop?
        if self.bern_drop():
            self.metrics.tx_dropped += 1
            return
        # delay = serialization + link latency + processing at receiver
        edge = self.edges[(node_id, hop)]
        ser_ms = int(1000.0 * tx.size_kB / max(1e-6, edge.bandwidth_kBps))
        prop_ms = self.sample_link_latency_ms(node_id, hop)
        proc_ms = self.sample_proc_ms(hop)
        total = ser_ms + prop_ms + proc_ms
        self.q.push(self.time_ms + total, "deliver_tx", dict(tx=tx, to=hop))

    def on_tx_arrival(self, node_id: int, tx: Transaction):
        node = self.nodes[node_id]
        if tx.tx_id in node.mempool:
            # already seen here; ignore
            return
        node.mempool[tx.tx_id] = tx
        if node_id == tx.dst:
            # "delivery" (account-model apply, simplified)
            self.nodes[tx.src].balance -= (tx.value + tx.fee)
            self.nodes[tx.dst].balance += tx.value
            # fees would go to a miner upon inclusion; we just burn or hold for now
            self.metrics.tx_delivered += 1
            # update EWMA of latency
            rtt = max(0, self.time_ms - tx.created_ms)
            alpha = 0.1
            self.metrics.tx_avg_end_to_end_ms = (1 - alpha) * self.metrics.tx_avg_end_to_end_ms + alpha * rtt
        else:
            # forward onward
            self.schedule_tx_forward(node_id, tx)

    # ---------- Blocks (optional, for "blocks in flight") ----------

    def mine_tick_and_broadcast(self):
        H = sum(n.hashrate_hs for n in self.nodes.values())
        if H <= 0:
            return
        for n in self.nodes.values():
            share = n.hashrate_hs / H
            p = min(0.25, share * (self.cfg.dt_ms / self.cfg.target_block_time_ms))
            if self.rand.bernoulli(p):
                blk = self.create_block(n.node_id)
                self.broadcast_block(n.node_id, blk)

    def create_block(self, miner_id: int) -> Block:
        miner = self.nodes[miner_id]
        parent_id = miner.head_id
        # include up to block_size_txs (naive validity)
        txs = list(miner.mempool.values())
        txs.sort(key=lambda t: t.fee, reverse=True)
        included = txs[: self.cfg.block_size_txs]
        blk = Block(block_id=next(self.block_seq), miner=miner_id, parent_id=parent_id, timestamp_ms=self.time_ms, txs=included)
        self.on_block_arrival(miner_id, blk)  # miner accepts immediately
        return blk

    def on_block_arrival(self, node_id: int, blk: Block):
        node = self.nodes[node_id]
        if blk.block_id in node.seen_blocks:
            return
        parent_ok = blk.parent_id in node.blocks
        node.blocks[blk.block_id] = blk
        node.seen_blocks.add(blk.block_id)
        h_parent = node.heights.get(blk.parent_id, -1)
        node.heights[blk.block_id] = h_parent + 1
        if node.head_id is None or node.heights[blk.block_id] > node.heights.get(node.head_id, -1):
            node.head_id = blk.block_id
            # apply coinbase + fees (simplified)
            fees = sum(t.fee for t in blk.txs)
            self.nodes[blk.miner].balance += (self.cfg.coinbase_reward + fees)
        # prune included txs from mempool
        for t in blk.txs:
            node.mempool.pop(t.tx_id, None)

    def broadcast_block(self, origin: int, blk: Block):
        for nb in self.nodes[origin].neighbors:
            # allow block drop? keep to tx only for now; blocks assumed reliable
            lat = self.sample_link_latency_ms(origin, nb)
            self.q.push(self.time_ms + lat, "deliver_block", dict(block=blk, to=nb))

    # ---------- Ticking ----------

    def tick(self):
        # inject new txs
        self.maybe_inject_tx()
        # mining (optional)
        self.mine_tick_and_broadcast()
        # deliver ready events
        for ev in self.q.pop_ready(self.time_ms):
            if ev.kind == "deliver_tx":
                self.on_tx_arrival(ev.payload["to"], ev.payload["tx"])
            elif ev.kind == "deliver_block":
                self.on_block_arrival(ev.payload["to"], ev.payload["block"])
        self.time_ms += self.cfg.dt_ms

    # ---------- Snapshots ----------

    def snapshot(self) -> NetworkState:
        return NetworkState(
            time_ms=self.time_ms,
            balances={i: n.balance for i, n in self.nodes.items()},
            hashrates={i: n.hashrate_hs for i, n in self.nodes.items()},
            latency_means={i: n.latency_mean_ms for i, n in self.nodes.items()},
            mempool_sizes={i: len(n.mempool) for i, n in self.nodes.items()},
            inflight_blocks=self.q.count_inflight_blocks(),
            inflight_txs=self.q.count_inflight_txs(),
        )

# Gym-like Environment (placeholders only)

class TxRoutingEnv:
    """A wrapper exposing obs/action/reward hooks for RL LATER.
    Right now: just defines shapes and accessors; no training logic.
    """
    def __init__(self, sim: Sim):
        self.sim = sim
        # Define observation / action spaces conceptually
        self.observation_spec = {
            "time_ms": "int",
            "balances": "Dict[node_id->float]",
            "latency_means": "Dict[node_id->float]",
            "mempool_sizes": "Dict[node_id->int]",
            "inflight_blocks": "int",
            "inflight_txs": "int",
        }
        # Action spec: mapping of {tx_id -> next_hop_node_id} per decision epoch
        self.action_spec = {"route_decisions": "Dict[int->int]"}

    def reset(self):
        # For later: rebuild Sim with same cfg/seed or new seed
        return self.sim.snapshot()

    def step(self, action: Optional[Dict[int, int]] = None):
        """For later: apply external routing decisions. Not used now."""
        # Placeholder: ignore actions now
        self.sim.tick()
        obs = self.sim.snapshot()
        reward = None  # <- DO NOT define yet
        done = self.sim.time_ms >= self.sim.cfg.sim_time_ms
        info = {}
        return obs, reward, done, info

def demo_run():
    cfg = SimConfig()
    sim = Sim(cfg)
    env = TxRoutingEnv(sim)
    obs = env.reset()
    # run without decisions (placeholder router handles forwarding)
    while sim.time_ms < cfg.sim_time_ms:
        obs, reward, done, info = env.step(action=None)
        if sim.time_ms % 5000 == 0:  # print every ~5s
            snap = sim.snapshot()
            print(
                f"t={snap.time_ms/1000:6.1f}s | Inflight blocks={snap.inflight_blocks} | "
                f"Inflight txs={snap.inflight_txs} | avg e2e={sim.metrics.tx_avg_end_to_end_ms:.1f} ms"
            )
        if done:
            break

if __name__ == "__main__":
    demo_run()
