# mini_chain_sim.py
# A minimal PoW blockchain network simulator with balances, hashrate, blocks-in-flight, and latency.
# Uses Gaussian (normal) for latency/jitter and Bernoulli for per-step events.

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
import heapq
import itertools
import time

# -----------------------------
# Data models
# -----------------------------
@dataclass
class Tx:
    tx_id: int
    src: int
    dst: int
    amount: float
    fee: float

@dataclass
class Block:
    block_id: int
    miner: int
    parent_id: Optional[int]
    timestamp_ms: int  # when mined (global sim time)
    txs: List[Tx] = field(default_factory=list)

# -----------------------------
# Node (miner/full node)
# -----------------------------
class Node:
    def __init__(self, node_id: int, init_balance: float, hashrate_hs: float,
                 latency_mean_ms: float, latency_std_ms: float):
        self.id = node_id
        self.balance = init_balance
        self.hashrate = hashrate_hs
        self.latency_mean = latency_mean_ms
        self.latency_std = latency_std_ms
        self.neighbors: List[int] = []  # full mesh by default; can customize
        self.mempool: Dict[int, Tx] = {}
        self.seen_blocks: set[int] = set()
        # simple chain view
        self.blocks: Dict[int, Block] = {}
        self.heights: Dict[int, int] = {}  # block_id -> height
        self.head_id: Optional[int] = None

    def add_neighbor(self, nid: int):
        if nid != self.id and nid not in self.neighbors:
            self.neighbors.append(nid)

    def chain_height(self) -> int:
        if self.head_id is None:
            return -1
        return self.heights.get(self.head_id, -1)

# -----------------------------
# Event queue
# -----------------------------
# events are tuples: (deliver_time_ms, seq, event_type, payload)
# event_type in {"deliver_block", "deliver_tx"}
class EventQueue:
    def __init__(self):
        self._pq: List[Tuple[int, int, str, dict]] = []
        self._seq = itertools.count()

    def push(self, when_ms: int, event_type: str, payload: dict):
        heapq.heappush(self._pq, (when_ms, next(self._seq), event_type, payload))

    def pop_ready(self, now_ms: int) -> List[Tuple[str, dict]]:
        out = []
        while self._pq and self._pq[0][0] <= now_ms:
            _, _, et, pl = heapq.heappop(self._pq)
            out.append((et, pl))
        return out

    def count_inflight_blocks(self) -> int:
        # Count pending deliver_block events
        return sum(1 for _, _, et, _ in self._pq if et == "deliver_block")

    def avg_scheduled_latency_ms(self) -> float:
        # Not exact per-block latency, but good for a summary: only for deliver_block events
        latencies = []
        now = int(time.time() * 1000)  # not the sim time; just for fallback
        for when_ms, _, et, pl in self._pq:
            if et == "deliver_block" and "scheduled_at" in pl:
                latencies.append(max(0, when_ms - pl["scheduled_at"]))
        return float(np.mean(latencies)) if latencies else 0.0

# -----------------------------
# Network simulator
# -----------------------------
class NetworkSim:
    def __init__(self,
                 n_nodes: int = 5,
                 target_block_time_ms: int = 30_000,
                 dt_ms: int = 100,                 # simulation tick
                 sim_time_ms: int = 120_000,       # total run time
                 tx_bernoulli_p: float = 0.1,      # per-tick chance to create a tx
                 tx_amount_mean: float = 1.0,
                 tx_fee_mean: float = 0.01,
                 block_size_txs: int = 500,
                 seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.dt_ms = dt_ms
        self.time_ms = 0
        self.end_time_ms = sim_time_ms
        self.target_block_time_ms = target_block_time_ms
        self.block_size_txs = block_size_txs
        self.tx_p = tx_bernoulli_p
        self.tx_amount_mean = tx_amount_mean
        self.tx_fee_mean = tx_fee_mean

        # Build nodes with varied hashrate/latency using Gaussian for realism
        self.nodes: Dict[int, Node] = {}
        base_hash = 1e9  # 1 GH/s baseline
        base_bal = 100.0
        for i in range(n_nodes):
            # Hashrate: lognormal-ish by exponentiating a normal for diversity
            hr = base_hash * float(np.exp(self.rng.normal(0.0, 0.4)))
            # Latency: normal clipped at >= 1 ms
            lat_mean = max(1.0, float(self.rng.normal(60.0, 20.0)))
            lat_std = max(5.0, lat_mean * 0.25)
            self.nodes[i] = Node(i, base_bal, hr, lat_mean, lat_std)

        # Fully connect the graph
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    self.nodes[i].add_neighbor(j)

        # Genesis
        self.block_seq = itertools.count(start=1)
        self.tx_seq = itertools.count(start=1)
        self.global_blocks: Dict[int, Block] = {}
        self.genesis = Block(block_id=0, miner=-1, parent_id=None, timestamp_ms=0, txs=[])
        for n in self.nodes.values():
            n.blocks[0] = self.genesis
            n.heights[0] = 0
            n.head_id = 0
            n.seen_blocks.add(0)

        self.q = EventQueue()
        self._last_stats_print = -999999

        # mining difficulty proxy: we use Bernoulli per tick scaled to target time
        self._difficulty_scale = 1.0  # conceptual placeholder

    # -------------------------
    # Sampling helpers
    # -------------------------
    def sample_latency(self, node_from: Node, node_to: Node) -> int:
        # Gaussian latency in ms, clipped
        mu = 0.5 * (node_from.latency_mean + node_to.latency_mean)
        sigma = 0.5 * (node_from.latency_std + node_to.latency_std)
        sample = self.rng.normal(mu, sigma)
        return int(max(1.0, sample))

    def maybe_create_random_tx(self):
        # Bernoulli: per tick, maybe create 1 transaction (keeps it simple)
        if self.rng.binomial(1, self.tx_p) == 0:
            return None
        # pick src != dst; bias src by balance (richer accounts more likely to spend)
        node_ids = list(self.nodes.keys())
        balances = np.array([max(0.0, self.nodes[i].balance) for i in node_ids], dtype=float)
        weights = balances + 1.0  # avoid zero-probability
        src = int(self.rng.choice(node_ids, p=weights / weights.sum()))
        dst = int(self.rng.choice([i for i in node_ids if i != src]))
        amount = max(0.001, float(self.rng.normal(self.tx_amount_mean, 0.25 * self.tx_amount_mean)))
        fee = max(0.0001, float(self.rng.normal(self.tx_fee_mean, 0.2 * self.tx_fee_mean)))
        # Only create if src can afford (soft check; final check at inclusion time)
        if self.nodes[src].balance < amount + fee:
            return None
        tx = Tx(tx_id=next(self.tx_seq), src=src, dst=dst, amount=amount, fee=fee)
        return tx

    def total_hashrate(self) -> float:
        return sum(n.hashrate for n in self.nodes.values())

    # -------------------------
    # Propagation
    # -------------------------
    def broadcast_block(self, origin_id: int, block: Block):
        for nid in self.nodes[origin_id].neighbors:
            delay = self.sample_latency(self.nodes[origin_id], self.nodes[nid])
            payload = dict(
                block=block,
                to=nid,
                scheduled_at=self.time_ms
            )
            self.q.push(self.time_ms + delay, "deliver_block", payload)

    def broadcast_tx(self, origin_id: int, tx: Tx):
        for nid in self.nodes[origin_id].neighbors:
            delay = self.sample_latency(self.nodes[origin_id], self.nodes[nid])
            self.q.push(self.time_ms + delay, "deliver_tx", dict(tx=tx, to=nid))

    def on_block_arrival(self, node: Node, block: Block):
        if block.block_id in node.seen_blocks:
            return
        # accept parent?
        parent_ok = (block.parent_id in node.blocks)
        if not parent_ok:
            # naive handling: we'll still store it but not advance head if parent missing
            node.blocks[block.block_id] = block
            node.heights[block.block_id] = node.heights.get(block.parent_id, -1) + 1
            node.seen_blocks.add(block.block_id)
            return

        # compute new height and potential reorg
        h_parent = node.heights[block.parent_id]
        h_new = h_parent + 1
        node.blocks[block.block_id] = block
        node.heights[block.block_id] = h_new
        node.seen_blocks.add(block.block_id)

        if node.head_id is None or h_new > node.heights.get(node.head_id, -1):
            node.head_id = block.block_id
            # apply coinbase/fees immediately for simplicity ONLY on first acceptance
            # (A fuller model would do UTXO + reorg-aware balance changes.)
            self.apply_block_rewards(node, block)

        # remove included txs from mempool
        for tx in block.txs:
            node.mempool.pop(tx.tx_id, None)

    def on_tx_arrival(self, node: Node, tx: Tx):
        if tx.tx_id in node.mempool:
            return
        node.mempool[tx.tx_id] = tx

    # -------------------------
    # Mining
    # -------------------------
    def mine_step(self):
        # Per tick Bernoulli for each node
        H_total = self.total_hashrate()
        if H_total <= 0:
            return []

        finders = []
        for node in self.nodes.values():
            share = node.hashrate / H_total
            # Bernoulli success prob scaled to match target_block_time
            p = min(0.25, share * (self.dt_ms / self.target_block_time_ms))
            if self.rng.binomial(1, p) == 1:
                finders.append(node.id)
        return finders

    def create_block(self, miner_id: int) -> Block:
        miner = self.nodes[miner_id]
        parent_id = miner.head_id
        # Select top-fee txs up to block size that are valid (src has balance)
        txs = list(miner.mempool.values())
        txs.sort(key=lambda t: t.fee, reverse=True)
        included = []
        gas = 0
        # naive validity check against miner's local balances (not UTXO accurate)
        local_balances = {i: self.nodes[i].balance for i in self.nodes}
        for tx in txs:
            if gas >= self.block_size_txs:
                break
            if local_balances.get(tx.src, 0.0) >= tx.amount + tx.fee:
                local_balances[tx.src] -= (tx.amount + tx.fee)
                local_balances[tx.dst] = local_balances.get(tx.dst, 0.0) + tx.amount
                included.append(tx)
                gas += 1

        block = Block(
            block_id=next(self.block_seq),
            miner=miner_id,
            parent_id=parent_id,
            timestamp_ms=self.time_ms,
            txs=included
        )
        # Miner immediately accepts own block
        self.on_block_arrival(miner, block)
        return block

    def apply_block_rewards(self, node: Node, block: Block):
        # naive economic model:
        COINBASE = 6.25   # fixed subsidy for demo
        fees = sum(tx.fee for tx in block.txs)
        self.nodes[block.miner].balance += (COINBASE + fees)

    # -------------------------
    # Ticking & stats
    # -------------------------
    def tick(self):
        # 1) Transactions creation at a random node
        tx = self.maybe_create_random_tx()
        if tx:
            # put into creator mempool and broadcast
            creator = int(tx.src)
            self.nodes[creator].mempool[tx.tx_id] = tx
            self.broadcast_tx(creator, tx)

        # 2) Mining
        winners = self.mine_step()
        # If multiple miners find blocks in same tick, forks happen
        for wid in winners:
            block = self.create_block(wid)
            self.global_blocks[block.block_id] = block
            self.broadcast_block(wid, block)

        # 3) Deliver ready events
        for et, payload in self.q.pop_ready(self.time_ms):
            if et == "deliver_block":
                node = self.nodes[payload["to"]]
                self.on_block_arrival(node, payload["block"])
            elif et == "deliver_tx":
                node = self.nodes[payload["to"]]
                self.on_tx_arrival(node, payload["tx"])

        self.time_ms += self.dt_ms

    def print_stats(self):
        now_s = self.time_ms / 1000.0
        if self.time_ms - self._last_stats_print < 5_000:
            return
        self._last_stats_print = self.time_ms

        total_bal = sum(n.balance for n in self.nodes.values())
        total_hash = self.total_hashrate()
        inflight_blocks = self.q.count_inflight_blocks()
        avg_lat = self.q.avg_scheduled_latency_ms()

        header = f"\n=== t={now_s:7.2f}s | BlocksInFlight={inflight_blocks} | AvgPropLatency≈{avg_lat:5.1f} ms ==="
        print(header)
        print(f"{'Node':>4} | {'Balance':>10} | {'%':>6} | {'Power (H/s)':>12} | {'%':>6} | {'Mempool':>7}")
        print("-"*64)
        for i, n in self.nodes.items():
            bal_pct = (n.balance / total_bal * 100.0) if total_bal > 0 else 0.0
            pow_pct = (n.hashrate / total_hash * 100.0) if total_hash > 0 else 0.0
            print(f"{i:4d} | {n.balance:10.3f} | {bal_pct:6.2f} | {n.hashrate:12.2f} | {pow_pct:6.2f} | {len(n.mempool):7d}")

    def run(self):
        while self.time_ms < self.end_time_ms:
            self.tick()
            self.print_stats()

# -----------------------------
# Demo / Config
# -----------------------------
if __name__ == "__main__":
    CONFIG = dict(
        n_nodes=6,
        target_block_time_ms=25_000,   # average ~25s per block across the network
        dt_ms=100,                     # 100ms per tick
        sim_time_ms=120_000,           # run 2 minutes
        tx_bernoulli_p=0.15,           # 15% chance to create one tx per tick
        tx_amount_mean=1.0,
        tx_fee_mean=0.02,
        block_size_txs=800,
        seed=1337
    )
    sim = NetworkSim(**CONFIG)
    sim.run()
