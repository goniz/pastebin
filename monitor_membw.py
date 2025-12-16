#!/usr/bin/env python3
"""
Memory Bandwidth and Cache Statistics Monitor for AMD EPYC Systems

This script monitors memory bandwidth and cache statistics on Linux systems
with AMD EPYC processors. It provides real-time visibility into:
- Per-NUMA node memory bandwidth estimation
- Cache miss rates and memory pressure indicators
- Top processes contributing to memory traffic
- Trends for key metrics

Designed for systems with 2 NUMA nodes and AMD EPYC Milan processors.

Usage:
    python3 monitor_membw.py [--interval SECONDS] [--top N]

Requirements:
    - Linux system with /proc and /sys filesystems
    - Python 3.6+
    - Optional: root access for perf-based metrics
"""

import argparse
import curses
import os
import signal
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

# Constants
PAGE_SIZE = 4096  # bytes
CACHE_LINE_SIZE = 64  # bytes for AMD EPYC
ZONE_PATTERN = re.compile(
    r"Node\s+(\d+),\s+zone\s+(\w+)(.*?)(?=Node\s+\d+,\s+zone|\Z)", re.DOTALL
)


@dataclass
class NumaStats:
    """NUMA node statistics from /proc/vmstat"""

    numa_hit: int = 0
    numa_miss: int = 0
    numa_foreign: int = 0
    numa_interleave: int = 0
    numa_local: int = 0
    numa_other: int = 0


@dataclass
class VmStats:
    """Virtual memory statistics from /proc/vmstat"""

    pgfault: int = 0
    pgmajfault: int = 0
    pgpgin: int = 0
    pgpgout: int = 0
    pswpin: int = 0
    pswpout: int = 0
    pgalloc_normal: int = 0
    pgalloc_dma: int = 0
    pgalloc_dma32: int = 0
    pgfree: int = 0
    pgactivate: int = 0
    pgdeactivate: int = 0
    pgreuse: int = 0
    numa_hit: int = 0
    numa_miss: int = 0
    numa_foreign: int = 0
    numa_local: int = 0
    numa_other: int = 0
    # Additional allocation stall metrics
    allocstall_normal: int = 0
    allocstall_dma: int = 0
    allocstall_dma32: int = 0
    allocstall_movable: int = 0
    # Compaction metrics (indicates memory pressure)
    compact_stall: int = 0
    compact_fail: int = 0
    compact_success: int = 0
    # Direct reclaim metrics
    pgsteal_kswapd: int = 0
    pgsteal_direct: int = 0
    pgscan_kswapd: int = 0
    pgscan_direct: int = 0


@dataclass
class ZoneInfo:
    """Per-zone memory allocator statistics from /proc/zoneinfo"""

    node_id: int
    zone_name: str
    pages_free: int = 0
    pages_min: int = 0
    pages_low: int = 0
    pages_high: int = 0
    managed: int = 0
    # Per-CPU pagelist (PCP) configuration
    pcp_count: int = 0  # Current pages in PCP across all CPUs
    pcp_high: int = 0  # High watermark per CPU
    pcp_batch: int = 0  # Batch size for refills
    pcp_count_list: List[int] = field(default_factory=list)  # Per-CPU counts


@dataclass
class PCPStats:
    """Per-CPU Pagelist statistics - key for zone lock contention analysis"""

    node_id: int
    zone_name: str
    high: int = 0  # High watermark (pages)
    batch: int = 0  # Batch size for zone->lock acquisitions
    cpu_counts: List[int] = field(default_factory=list)  # Per-CPU current counts
    total_count: int = 0  # Sum of all CPU counts
    num_cpus: int = 0  # Number of CPUs with PCP for this zone


@dataclass
class ProcessMemStats:
    """Per-process memory statistics"""

    pid: int
    name: str
    rss: int = 0  # Resident Set Size in KB
    vms: int = 0  # Virtual Memory Size in KB
    shared: int = 0  # Shared memory in KB
    minor_faults: int = 0
    major_faults: int = 0
    # Child fault stats (cumulative for waited-for children)
    cmin_flt: int = 0  # Child minor faults
    cmaj_flt: int = 0  # Child major faults
    read_bytes: int = 0
    write_bytes: int = 0
    # Additional memory metrics from /proc/[pid]/status
    rss_anon: int = 0  # Anonymous RSS in KB
    rss_file: int = 0  # File-backed RSS in KB
    rss_shmem: int = 0  # Shared memory RSS in KB
    vm_data: int = 0  # Data segment size in KB
    vm_stk: int = 0  # Stack size in KB
    vm_swap: int = 0  # Swapped-out virtual memory in KB
    # Scheduling/timing for allocation rate estimation
    utime: int = 0  # User time in clock ticks
    stime: int = 0  # System time in clock ticks
    # I/O metrics from /proc/[pid]/io
    rchar: int = 0  # Characters read (includes page cache)
    wchar: int = 0  # Characters written
    syscr: int = 0  # Read syscalls
    syscw: int = 0  # Write syscalls
    cancelled_write_bytes: int = 0  # Cancelled write bytes
    # NUMA metrics from /proc/[pid]/numa_maps
    numa_pages: Dict[int, int] = field(default_factory=dict)  # node_id -> pages
    numa_total_pages: int = 0  # Total pages across all nodes
    numa_anon_pages: Dict[int, int] = field(
        default_factory=dict
    )  # Anonymous pages per node
    numa_file_pages: Dict[int, int] = field(
        default_factory=dict
    )  # File-backed pages per node
    numa_heap_pages: Dict[int, int] = field(default_factory=dict)  # Heap pages per node
    numa_stack_pages: Dict[int, int] = field(
        default_factory=dict
    )  # Stack pages per node
    count: int = 1  # Number of merged processes (for --merge mode)


@dataclass
class NodeMemInfo:
    """Per-NUMA node memory info"""

    node_id: int
    mem_total: int = 0
    mem_free: int = 0
    mem_used: int = 0
    active: int = 0
    inactive: int = 0
    file_pages: int = 0
    anon_pages: int = 0
    dirty: int = 0
    writeback: int = 0


@dataclass
class TrendData:
    """Stores historical data for trend calculation"""

    values: List[float] = field(default_factory=list)
    max_samples: int = 10

    def add(self, value: float):
        self.values.append(value)
        if len(self.values) > self.max_samples:
            self.values.pop(0)

    def trend(self) -> str:
        """Returns trend indicator: ↑ ↓ → or ?"""
        if len(self.values) < 3:
            return "?"
        recent = sum(self.values[-3:]) / 3
        older = sum(self.values[:3]) / 3
        diff = recent - older
        threshold = max(abs(older) * 0.05, 1)  # 5% threshold
        if diff > threshold:
            return "↑"
        elif diff < -threshold:
            return "↓"
        return "→"


class MemoryBandwidthMonitor:
    """Main monitoring class for memory bandwidth and cache statistics"""

    def __init__(
        self, interval: float = 1.0, top_n: int = 10, merge_by_name: bool = False
    ):
        self.interval = interval
        self.top_n = top_n
        self.merge_by_name = merge_by_name
        self.prev_vmstats: Optional[VmStats] = None
        self.prev_time: float = 0
        self.prev_proc_stats: Dict[int, ProcessMemStats] = {}
        self.numa_nodes = self._detect_numa_nodes()
        self.cpu_count = os.cpu_count() or 1
        self.has_perf = self._check_perf_available()
        self.perf_process: Optional[subprocess.Popen] = None

        # Trend trackers
        self.trends = {
            "mem_bandwidth": TrendData(),
            "numa_local": TrendData(),
            "numa_remote": TrendData(),
            "page_faults": TrendData(),
            "major_faults": TrendData(),
        }

        # Per-node trends
        self.node_trends = {node: TrendData() for node in self.numa_nodes}

        # Zone/PCP tracking for lock contention analysis
        self.prev_zone_stats: Dict[str, ZoneInfo] = {}
        self.trends["zone_lock_pressure"] = TrendData()
        self.trends["pcp_refill_rate"] = TrendData()

    def _detect_numa_nodes(self) -> List[int]:
        """Detect available NUMA nodes"""
        nodes = []
        numa_path = Path("/sys/devices/system/node")
        if numa_path.exists():
            for entry in numa_path.iterdir():
                if entry.name.startswith("node") and entry.name[4:].isdigit():
                    nodes.append(int(entry.name[4:]))
        return sorted(nodes) if nodes else [0]

    def _check_perf_available(self) -> bool:
        """Check if perf is available and we have permissions"""
        try:
            result = subprocess.run(["perf", "list"], capture_output=True, timeout=5)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):
            return False

    def read_vmstat(self) -> VmStats:
        """Read /proc/vmstat and parse relevant counters"""
        stats = VmStats()
        try:
            with open("/proc/vmstat", "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        key, value = parts[0], int(parts[1])
                        if hasattr(stats, key):
                            setattr(stats, key, value)
        except (IOError, ValueError) as e:
            pass  # Return default stats on error
        return stats

    def read_numa_node_meminfo(self, node_id: int) -> NodeMemInfo:
        """Read per-NUMA node memory info from sysfs"""
        info = NodeMemInfo(node_id=node_id)
        meminfo_path = f"/sys/devices/system/node/node{node_id}/meminfo"

        try:
            with open(meminfo_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        # Format: "Node X KeyName: Value kB"
                        key = parts[2].rstrip(":")
                        value = int(parts[3])

                        key_map = {
                            "MemTotal": "mem_total",
                            "MemFree": "mem_free",
                            "MemUsed": "mem_used",
                            "Active": "active",
                            "Inactive": "inactive",
                            "FilePages": "file_pages",
                            "AnonPages": "anon_pages",
                            "Dirty": "dirty",
                            "Writeback": "writeback",
                        }

                        if key in key_map:
                            setattr(info, key_map[key], value)
        except (IOError, ValueError, IndexError):
            pass

        return info

    def read_numa_stats(self) -> Dict[int, Dict[str, int]]:
        """Read per-node NUMA statistics from /sys"""
        node_stats = {}

        for node_id in self.numa_nodes:
            stats = {}
            numastat_path = f"/sys/devices/system/node/node{node_id}/numastat"

            try:
                with open(numastat_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            stats[parts[0]] = int(parts[1])
            except (IOError, ValueError):
                pass

            node_stats[node_id] = stats

        return node_stats

    def read_zoneinfo(self) -> Dict[str, ZoneInfo]:
        """Read /proc/zoneinfo for zone and PCP statistics

        This is critical for understanding zone lock contention:
        - PCP high/batch determine how often zone->lock is acquired
        - Zone free pages indicate allocation pressure
        """
        zones = {}

        try:
            with open("/proc/zoneinfo", "r") as f:
                content = f.read()
        except IOError:
            return zones

        # Parse each zone section
        # Format: "Node N, zone   ZoneName"

        for match in ZONE_PATTERN.finditer(content):
            node_id = int(match.group(1))
            zone_name = match.group(2)
            zone_content = match.group(3)

            key = f"node{node_id}_{zone_name}"
            zone = ZoneInfo(node_id=node_id, zone_name=zone_name)

            # Parse zone statistics
            for line in zone_content.split("\n"):
                line = line.strip()
                parts = line.split()

                if len(parts) >= 2:
                    field_name = parts[0]
                    try:
                        value = int(parts[1])

                        if field_name == "free":
                            zone.pages_free = value
                        elif field_name == "min":
                            zone.pages_min = value
                        elif field_name == "low":
                            zone.pages_low = value
                        elif field_name == "high":
                            # This is zone high watermark, not PCP high
                            zone.pages_high = value
                        elif field_name == "managed":
                            zone.managed = value
                    except ValueError:
                        pass

                # Parse PCP (pagesets) section
                # Format after "pagesets" line:
                #   cpu: N
                #     count: X
                #     high:  Y
                #     batch: Z
                if "high:" in line and "pagesets" not in line:
                    try:
                        pcp_high = int(line.split(":")[1].strip())
                        if zone.pcp_high == 0:  # Take first (they should all be same)
                            zone.pcp_high = pcp_high
                    except (ValueError, IndexError):
                        pass

                if "batch:" in line:
                    try:
                        pcp_batch = int(line.split(":")[1].strip())
                        if zone.pcp_batch == 0:
                            zone.pcp_batch = pcp_batch
                    except (ValueError, IndexError):
                        pass

                if "count:" in line and "nr_" not in line:
                    try:
                        count = int(line.split(":")[1].strip())
                        zone.pcp_count_list.append(count)
                        zone.pcp_count += count
                    except (ValueError, IndexError):
                        pass

            zones[key] = zone

        return zones

    def read_pcp_stats(self) -> Dict[int, PCPStats]:
        """Read Per-CPU Pagelist statistics per NUMA node

        Returns aggregated PCP stats per node for the Normal zone
        (which is where most allocations happen)
        """
        pcp_stats = {}
        zones = self.read_zoneinfo()

        for key, zone in zones.items():
            if zone.zone_name != "Normal":
                continue

            pcp = PCPStats(
                node_id=zone.node_id,
                zone_name=zone.zone_name,
                high=zone.pcp_high,
                batch=zone.pcp_batch,
                cpu_counts=zone.pcp_count_list.copy(),
                total_count=zone.pcp_count,
                num_cpus=len(zone.pcp_count_list),
            )
            pcp_stats[zone.node_id] = pcp

        return pcp_stats

    def calculate_zone_lock_metrics(
        self, rates: Dict, pcp_stats: Dict[int, PCPStats]
    ) -> Dict:
        """Calculate zone lock contention indicators

        Key metrics:
        - Estimated zone lock acquisitions per second
        - PCP pressure (how often PCP is depleted)
        - Lock contention probability estimate

        Returns metrics for the node with the highest pressure
        """
        metrics = {}

        alloc_rate = rates.get("alloc_pages_s", 0)

        # Calculate metrics for all nodes and pick the worst case (highest pressure)
        max_pressure = -1.0
        worst_node_metrics = None

        # If no PCP stats, return empty/zero metrics
        if not pcp_stats:
            metrics["zone_lock_acq_per_sec"] = 0
            metrics["pcp_utilization_pct"] = 0
            metrics["pcp_exhaust_time_ms"] = float("inf")
            metrics["lock_contention_pressure"] = 0
            metrics["pcp_high"] = 0
            metrics["pcp_batch"] = 0
            metrics["pcp_total_cached"] = 0
            metrics["worst_node"] = -1
            metrics["allocstall_rate"] = rates.get("allocstall_s", 0)
            return metrics

        for node_id, node_pcp in pcp_stats.items():
            node_metrics = {}

            if node_pcp.batch > 0:
                # Estimate zone lock acquisitions per second
                # Each batch refill requires acquiring zone->lock
                # Distribute alloc rate roughly by node size or assume uniform if unknown
                # For worst-case analysis, assume this node takes a proportional share
                # or simplified: alloc_rate / num_nodes
                node_alloc_rate = alloc_rate / max(1, len(pcp_stats))

                estimated_lock_acq = node_alloc_rate / node_pcp.batch
                node_metrics["zone_lock_acq_per_sec"] = estimated_lock_acq

                # PCP utilization - how full are the PCPs on average?
                if node_pcp.high > 0 and node_pcp.num_cpus > 0:
                    avg_pcp_count = node_pcp.total_count / node_pcp.num_cpus
                    node_metrics["pcp_utilization_pct"] = (
                        avg_pcp_count / node_pcp.high
                    ) * 100
                else:
                    node_metrics["pcp_utilization_pct"] = 0

                # Time to exhaust PCP per CPU (assuming uniform distribution)
                cpus_on_node = node_pcp.num_cpus if node_pcp.num_cpus > 0 else 64
                per_cpu_alloc_rate = node_alloc_rate / cpus_on_node
                if per_cpu_alloc_rate > 0 and node_pcp.high > 0:
                    node_metrics["pcp_exhaust_time_ms"] = (
                        node_pcp.high / per_cpu_alloc_rate
                    ) * 1000
                else:
                    node_metrics["pcp_exhaust_time_ms"] = float("inf")

                # Lock contention pressure indicator (0-100)
                # Higher value = more likely to have contention
                # Based on: lock_acq_rate * estimated_hold_time * num_cpus
                lock_hold_time_ns = 300  # Rough estimate
                lock_busy_fraction = estimated_lock_acq * (lock_hold_time_ns / 1e9)
                node_metrics["lock_contention_pressure"] = min(
                    100, lock_busy_fraction * 100 * cpus_on_node
                )

                node_metrics["pcp_high"] = node_pcp.high
                node_metrics["pcp_batch"] = node_pcp.batch
                node_metrics["pcp_total_cached"] = node_pcp.total_count
                node_metrics["worst_node"] = node_id
            else:
                node_metrics["zone_lock_acq_per_sec"] = 0
                node_metrics["pcp_utilization_pct"] = 0
                node_metrics["pcp_exhaust_time_ms"] = float("inf")
                node_metrics["lock_contention_pressure"] = 0
                node_metrics["pcp_high"] = 0
                node_metrics["pcp_batch"] = 0
                node_metrics["pcp_total_cached"] = 0
                node_metrics["worst_node"] = node_id

            # Track max pressure
            pressure = node_metrics.get("lock_contention_pressure", 0)
            if pressure > max_pressure:
                max_pressure = pressure
                worst_node_metrics = node_metrics

        # Use metrics from the node with worst contention
        if worst_node_metrics:
            metrics.update(worst_node_metrics)

        # Allocation stall rate (direct indicator of contention/pressure)
        metrics["allocstall_rate"] = rates.get("allocstall_s", 0)

        return metrics

    def get_process_mem_stats(self) -> List[ProcessMemStats]:
        """Get memory statistics for all processes"""
        processes = []

        try:
            for entry in os.listdir("/proc"):
                if not entry.isdigit():
                    continue

                pid = int(entry)
                try:
                    proc_stats = self._read_process_stats(pid)
                    if proc_stats:
                        processes.append(proc_stats)
                except (
                    IOError,
                    PermissionError,
                    ProcessLookupError,
                    FileNotFoundError,
                ):
                    continue
        except (IOError, PermissionError):
            pass

        return processes

    def _read_process_stats(self, pid: int) -> Optional[ProcessMemStats]:
        """Read stats for a single process (cheap metrics only)"""
        proc_path = f"/proc/{pid}"

        # Read process name
        try:
            with open(f"{proc_path}/comm", "r") as f:
                name = f.read().strip()
        except (IOError, PermissionError, FileNotFoundError):
            return None

        stats = ProcessMemStats(pid=pid, name=name)

        # Read statm for memory sizes
        try:
            with open(f"{proc_path}/statm", "r") as f:
                parts = f.read().strip().split()
                if len(parts) >= 3:
                    page_size_kb = PAGE_SIZE // 1024
                    stats.vms = int(parts[0]) * page_size_kb
                    stats.rss = int(parts[1]) * page_size_kb
                    stats.shared = int(parts[2]) * page_size_kb
        except (IOError, PermissionError, FileNotFoundError):
            pass

        # Read stat for page faults and timing
        # Fields from proc(5) man page after (comm) state:
        # 0:state 1:ppid 2:pgrp 3:session 4:tty_nr 5:tpgid 6:flags
        # 7:minflt 8:cminflt 9:majflt 10:cmajflt 11:utime 12:stime
        try:
            with open(f"{proc_path}/stat", "r") as f:
                content = f.read()
                # Handle process names with spaces/parentheses
                start = content.rfind(")")
                if start > 0:
                    fields = content[start + 2 :].split()
                    if len(fields) >= 13:
                        stats.minor_faults = int(fields[7])
                        stats.cmin_flt = int(fields[8])
                        stats.major_faults = int(fields[9])
                        stats.cmaj_flt = int(fields[10])
                        stats.utime = int(fields[11])
                        stats.stime = int(fields[12])
        except (IOError, PermissionError, ValueError, FileNotFoundError):
            pass

        # Read status for detailed memory breakdown
        try:
            with open(f"{proc_path}/status", "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        key = parts[0].rstrip(":")
                        try:
                            value = int(parts[1])
                            if key == "RssAnon":
                                stats.rss_anon = value
                            elif key == "RssFile":
                                stats.rss_file = value
                            elif key == "RssShmem":
                                stats.rss_shmem = value
                            elif key == "VmData":
                                stats.vm_data = value
                            elif key == "VmStk":
                                stats.vm_stk = value
                            elif key == "VmSwap":
                                stats.vm_swap = value
                        except (ValueError, IndexError):
                            pass
        except (IOError, PermissionError, FileNotFoundError):
            pass

        # Read io stats if available (more detailed)
        try:
            with open(f"{proc_path}/io", "r") as f:
                for line in f:
                    parts = line.strip().split(":")
                    if len(parts) == 2:
                        key, value = parts[0].strip(), int(parts[1].strip())
                        if key == "read_bytes":
                            stats.read_bytes = value
                        elif key == "write_bytes":
                            stats.write_bytes = value
                        elif key == "rchar":
                            stats.rchar = value
                        elif key == "wchar":
                            stats.wchar = value
                        elif key == "syscr":
                            stats.syscr = value
                        elif key == "syscw":
                            stats.syscw = value
                        elif key == "cancelled_write_bytes":
                            stats.cancelled_write_bytes = value
        except (IOError, PermissionError, FileNotFoundError):
            pass

        return stats

    def _enrich_process_numa_stats(self, stats: ProcessMemStats):
        """Read expensive NUMA maps only for top processes"""
        proc_path = f"/proc/{stats.pid}"

        # Read numa_maps for per-process NUMA distribution
        # Format: address policy mapping_details
        # Example: 7f1234000000 default file=/lib/x86_64-linux-gnu/libc.so anon=1 dirty=1 N0=100 N1=50
        try:
            with open(f"{proc_path}/numa_maps", "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue

                    # Determine if this is heap, stack, anon, or file mapping
                    is_heap = "heap" in line
                    is_stack = "stack" in line
                    is_anon = "anon=" in line

                    # Parse node page counts (N0=xxx N1=xxx format)
                    for part in parts:
                        if part.startswith("N") and "=" in part:
                            try:
                                node_str, count_str = part.split("=")
                                node_id = int(node_str[1:])  # Remove 'N' prefix
                                page_count = int(count_str)

                                # Aggregate total pages per node
                                stats.numa_pages[node_id] = (
                                    stats.numa_pages.get(node_id, 0) + page_count
                                )
                                stats.numa_total_pages += page_count

                                # Categorize by mapping type
                                if is_anon or is_heap:
                                    stats.numa_anon_pages[node_id] = (
                                        stats.numa_anon_pages.get(node_id, 0)
                                        + page_count
                                    )
                                if is_heap:
                                    stats.numa_heap_pages[node_id] = (
                                        stats.numa_heap_pages.get(node_id, 0)
                                        + page_count
                                    )
                                if is_stack:
                                    stats.numa_stack_pages[node_id] = (
                                        stats.numa_stack_pages.get(node_id, 0)
                                        + page_count
                                    )
                                if not is_anon and not is_heap and not is_stack:
                                    stats.numa_file_pages[node_id] = (
                                        stats.numa_file_pages.get(node_id, 0)
                                        + page_count
                                    )
                            except (ValueError, IndexError):
                                pass
        except (IOError, PermissionError, FileNotFoundError):
            pass

    def calculate_bandwidth_rates(
        self, current: VmStats, prev: VmStats, elapsed: float
    ) -> Dict[str, float]:
        """Calculate memory bandwidth rates from vmstat deltas"""
        if elapsed <= 0:
            return {}

        rates = {}

        # Page activity based bandwidth estimation
        # Each page operation implies memory transfer
        pgpgin_delta = current.pgpgin - prev.pgpgin
        pgpgout_delta = current.pgpgout - prev.pgpgout

        # Convert to MB/s (pages are in 1KB blocks for pgpgin/pgpgout)
        rates["read_mb_s"] = (pgpgin_delta / 1024) / elapsed
        rates["write_mb_s"] = (pgpgout_delta / 1024) / elapsed
        rates["total_mb_s"] = rates["read_mb_s"] + rates["write_mb_s"]

        # Page fault rates
        rates["minor_faults_s"] = (current.pgfault - prev.pgfault) / elapsed
        rates["major_faults_s"] = (current.pgmajfault - prev.pgmajfault) / elapsed

        # Allocation rates (pages/sec)
        alloc_delta = (
            (current.pgalloc_normal - prev.pgalloc_normal)
            + (current.pgalloc_dma - prev.pgalloc_dma)
            + (current.pgalloc_dma32 - prev.pgalloc_dma32)
        )
        rates["alloc_pages_s"] = alloc_delta / elapsed
        rates["free_pages_s"] = (current.pgfree - prev.pgfree) / elapsed

        # NUMA rates
        rates["numa_local_s"] = (current.numa_local - prev.numa_local) / elapsed
        rates["numa_other_s"] = (current.numa_other - prev.numa_other) / elapsed
        rates["numa_hit_s"] = (current.numa_hit - prev.numa_hit) / elapsed
        rates["numa_miss_s"] = (current.numa_miss - prev.numa_miss) / elapsed

        # Calculate NUMA locality percentage
        total_numa = rates["numa_local_s"] + rates["numa_other_s"]
        if total_numa > 0:
            rates["numa_local_pct"] = (rates["numa_local_s"] / total_numa) * 100
        else:
            rates["numa_local_pct"] = 100.0

        # Memory pressure indicator (based on swap and major faults)
        swap_activity = (
            (current.pswpin - prev.pswpin) + (current.pswpout - prev.pswpout)
        ) / elapsed
        rates["swap_activity_s"] = swap_activity
        rates["memory_pressure"] = min(
            100, rates["major_faults_s"] + swap_activity * 10
        )

        # Allocation stall rates (direct indicators of zone lock contention/memory pressure)
        allocstall_delta = (
            (current.allocstall_normal - prev.allocstall_normal)
            + (current.allocstall_dma - prev.allocstall_dma)
            + (current.allocstall_dma32 - prev.allocstall_dma32)
            + (current.allocstall_movable - prev.allocstall_movable)
        )
        rates["allocstall_s"] = allocstall_delta / elapsed

        # Compaction rates (indicates memory fragmentation pressure)
        rates["compact_stall_s"] = (
            current.compact_stall - prev.compact_stall
        ) / elapsed

        # Direct reclaim rates (indicates memory pressure forcing synchronous reclaim)
        rates["pgsteal_direct_s"] = (
            current.pgsteal_direct - prev.pgsteal_direct
        ) / elapsed
        rates["pgscan_direct_s"] = (
            current.pgscan_direct - prev.pgscan_direct
        ) / elapsed

        return rates

    def calculate_node_bandwidth(
        self,
        current_stats: Dict[int, Dict[str, int]],
        prev_stats: Dict[int, Dict[str, int]],
        elapsed: float,
    ) -> Dict[int, Dict[str, float]]:
        """Calculate per-NUMA node bandwidth estimates"""
        node_rates = {}

        for node_id in self.numa_nodes:
            curr = current_stats.get(node_id, {})
            prev = prev_stats.get(node_id, {})

            rates = {}
            for key in [
                "numa_hit",
                "numa_miss",
                "numa_foreign",
                "local_node",
                "other_node",
            ]:
                curr_val = curr.get(key, 0)
                prev_val = prev.get(key, 0)
                if elapsed > 0:
                    rates[f"{key}_s"] = (curr_val - prev_val) / elapsed
                else:
                    rates[f"{key}_s"] = 0

            # Estimate bandwidth: each page access is PAGE_SIZE bytes
            local_accesses = rates.get("local_node_s", rates.get("numa_hit_s", 0))
            remote_accesses = rates.get("other_node_s", rates.get("numa_miss_s", 0))

            rates["local_mb_s"] = (local_accesses * PAGE_SIZE) / (1024 * 1024)
            rates["remote_mb_s"] = (remote_accesses * PAGE_SIZE) / (1024 * 1024)
            rates["total_mb_s"] = rates["local_mb_s"] + rates["remote_mb_s"]

            node_rates[node_id] = rates

        return node_rates

    def merge_processes_by_name(
        self, processes: List[ProcessMemStats]
    ) -> List[ProcessMemStats]:
        """Merge processes with the same name, aggregating their statistics"""
        merged: Dict[str, ProcessMemStats] = {}

        for proc in processes:
            if proc.name in merged:
                # Aggregate stats into existing entry
                existing = merged[proc.name]
                existing.rss += proc.rss
                existing.vms += proc.vms
                existing.shared += proc.shared
                existing.minor_faults += proc.minor_faults
                existing.major_faults += proc.major_faults
                existing.cmin_flt += proc.cmin_flt
                existing.cmaj_flt += proc.cmaj_flt
                existing.read_bytes += proc.read_bytes
                existing.write_bytes += proc.write_bytes
                existing.rss_anon += proc.rss_anon
                existing.rss_file += proc.rss_file
                existing.rss_shmem += proc.rss_shmem
                existing.vm_data += proc.vm_data
                existing.vm_stk += proc.vm_stk
                existing.vm_swap += proc.vm_swap
                existing.utime += proc.utime
                existing.stime += proc.stime
                existing.rchar += proc.rchar
                existing.wchar += proc.wchar
                existing.syscr += proc.syscr
                existing.syscw += proc.syscw
                existing.cancelled_write_bytes += proc.cancelled_write_bytes
                # Merge NUMA stats
                for node_id, pages in proc.numa_pages.items():
                    existing.numa_pages[node_id] = (
                        existing.numa_pages.get(node_id, 0) + pages
                    )
                existing.numa_total_pages += proc.numa_total_pages
                for node_id, pages in proc.numa_anon_pages.items():
                    existing.numa_anon_pages[node_id] = (
                        existing.numa_anon_pages.get(node_id, 0) + pages
                    )
                for node_id, pages in proc.numa_file_pages.items():
                    existing.numa_file_pages[node_id] = (
                        existing.numa_file_pages.get(node_id, 0) + pages
                    )
                for node_id, pages in proc.numa_heap_pages.items():
                    existing.numa_heap_pages[node_id] = (
                        existing.numa_heap_pages.get(node_id, 0) + pages
                    )
                for node_id, pages in proc.numa_stack_pages.items():
                    existing.numa_stack_pages[node_id] = (
                        existing.numa_stack_pages.get(node_id, 0) + pages
                    )
                existing.count += 1
            else:
                # Create a new merged entry (use negative pid to indicate merged)
                merged[proc.name] = ProcessMemStats(
                    pid=-1,  # Indicate this is a merged entry
                    name=proc.name,
                    rss=proc.rss,
                    vms=proc.vms,
                    shared=proc.shared,
                    minor_faults=proc.minor_faults,
                    major_faults=proc.major_faults,
                    cmin_flt=proc.cmin_flt,
                    cmaj_flt=proc.cmaj_flt,
                    read_bytes=proc.read_bytes,
                    write_bytes=proc.write_bytes,
                    rss_anon=proc.rss_anon,
                    rss_file=proc.rss_file,
                    rss_shmem=proc.rss_shmem,
                    vm_data=proc.vm_data,
                    vm_stk=proc.vm_stk,
                    vm_swap=proc.vm_swap,
                    utime=proc.utime,
                    stime=proc.stime,
                    rchar=proc.rchar,
                    wchar=proc.wchar,
                    syscr=proc.syscr,
                    syscw=proc.syscw,
                    cancelled_write_bytes=proc.cancelled_write_bytes,
                    numa_pages=proc.numa_pages.copy(),
                    numa_total_pages=proc.numa_total_pages,
                    numa_anon_pages=proc.numa_anon_pages.copy(),
                    numa_file_pages=proc.numa_file_pages.copy(),
                    numa_heap_pages=proc.numa_heap_pages.copy(),
                    numa_stack_pages=proc.numa_stack_pages.copy(),
                    count=1,
                )

        return list(merged.values())

    def get_top_memory_processes(
        self, processes: List[ProcessMemStats], prev_stats: Dict, elapsed: float
    ) -> List[Tuple[ProcessMemStats, Dict]]:
        """Get top processes by memory activity

        Args:
            processes: List of current process stats
            prev_stats: Previous stats dict (keyed by pid or name depending on merge mode)
            elapsed: Time elapsed since last sample
        """
        process_rates = []

        # Merge processes by name if enabled
        if self.merge_by_name:
            processes = self.merge_processes_by_name(processes)

        for proc in processes:
            rates = {}

            # For merged processes, look up by name; for individual, by pid
            if self.merge_by_name:
                prev = prev_stats.get(proc.name)
            else:
                prev = prev_stats.get(proc.pid)

            if prev and elapsed > 0:
                # Calculate fault rates
                rates["minor_faults_s"] = (
                    max(0, proc.minor_faults - prev.minor_faults) / elapsed
                )
                rates["major_faults_s"] = (
                    max(0, proc.major_faults - prev.major_faults) / elapsed
                )
                # Child fault rates (for processes that spawn children)
                rates["cmin_flt_s"] = max(0, proc.cmin_flt - prev.cmin_flt) / elapsed
                rates["cmaj_flt_s"] = max(0, proc.cmaj_flt - prev.cmaj_flt) / elapsed
                # Total fault rates (self + children)
                rates["total_minor_s"] = rates["minor_faults_s"] + rates["cmin_flt_s"]
                rates["total_major_s"] = rates["major_faults_s"] + rates["cmaj_flt_s"]

                # Estimated page allocation rate (minor faults often indicate new allocations)
                # Each minor fault typically means a new page was allocated
                rates["alloc_pages_s"] = rates["minor_faults_s"]  # Approximation
                rates["alloc_mb_s"] = (rates["alloc_pages_s"] * PAGE_SIZE) / (
                    1024 * 1024
                )

                # I/O rates (physical disk)
                rates["io_read_mb_s"] = max(0, proc.read_bytes - prev.read_bytes) / (
                    1024 * 1024 * elapsed
                )
                rates["io_write_mb_s"] = max(0, proc.write_bytes - prev.write_bytes) / (
                    1024 * 1024 * elapsed
                )

                # Virtual I/O rates (includes page cache)
                rates["vio_read_mb_s"] = max(0, proc.rchar - prev.rchar) / (
                    1024 * 1024 * elapsed
                )
                rates["vio_write_mb_s"] = max(0, proc.wchar - prev.wchar) / (
                    1024 * 1024 * elapsed
                )

                # Syscall rates
                rates["syscr_s"] = max(0, proc.syscr - prev.syscr) / elapsed
                rates["syscw_s"] = max(0, proc.syscw - prev.syscw) / elapsed
                rates["syscall_total_s"] = rates["syscr_s"] + rates["syscw_s"]

                # RSS change rate (positive = growing, negative = shrinking)
                rates["rss_change_kb_s"] = (proc.rss - prev.rss) / elapsed
                rates["rss_anon_change_s"] = (proc.rss_anon - prev.rss_anon) / elapsed

                # CPU time rate (for context)
                hz = os.sysconf("SC_CLK_TCK")  # Usually 100
                rates["cpu_user_pct"] = (
                    max(0, (proc.utime - prev.utime) / hz) / elapsed * 100
                )
                rates["cpu_sys_pct"] = (
                    max(0, (proc.stime - prev.stime) / hz) / elapsed * 100
                )
                rates["cpu_total_pct"] = rates["cpu_user_pct"] + rates["cpu_sys_pct"]
            else:
                rates["minor_faults_s"] = 0
                rates["major_faults_s"] = 0
                rates["cmin_flt_s"] = 0
                rates["cmaj_flt_s"] = 0
                rates["total_minor_s"] = 0
                rates["total_major_s"] = 0
                rates["alloc_pages_s"] = 0
                rates["alloc_mb_s"] = 0
                rates["io_read_mb_s"] = 0
                rates["io_write_mb_s"] = 0
                rates["vio_read_mb_s"] = 0
                rates["vio_write_mb_s"] = 0
                rates["syscr_s"] = 0
                rates["syscw_s"] = 0
                rates["syscall_total_s"] = 0
                rates["rss_change_kb_s"] = 0
                rates["rss_anon_change_s"] = 0
                rates["cpu_user_pct"] = 0
                rates["cpu_sys_pct"] = 0
                rates["cpu_total_pct"] = 0

            # Calculate NUMA locality from numa_maps data
            # NOTE: detailed numa_maps data is now loaded lazily only for top processes
            # so initially this will be empty until the second pass
            if proc.numa_total_pages > 0:
                # Find which node has the most pages (primary node)
                primary_node = (
                    max(proc.numa_pages.keys(), key=lambda n: proc.numa_pages.get(n, 0))
                    if proc.numa_pages
                    else 0
                )
                primary_pages = proc.numa_pages.get(primary_node, 0)
                rates["numa_local_pct"] = (primary_pages / proc.numa_total_pages) * 100
                rates["numa_primary_node"] = primary_node

                # Calculate distribution string (e.g., "N0:80% N1:20%")
                numa_dist = []
                for node_id in sorted(proc.numa_pages.keys()):
                    pct = (proc.numa_pages[node_id] / proc.numa_total_pages) * 100
                    if pct >= 1:  # Only show nodes with >= 1%
                        numa_dist.append(f"N{node_id}:{pct:.0f}%")
                rates["numa_dist_str"] = " ".join(numa_dist) if numa_dist else "N/A"

                # Memory in MB per node
                rates["numa_mb"] = {
                    n: (p * PAGE_SIZE) / (1024 * 1024)
                    for n, p in proc.numa_pages.items()
                }
            else:
                rates["numa_local_pct"] = 0
                rates["numa_primary_node"] = -1
                rates["numa_dist_str"] = "N/A"
                rates["numa_mb"] = {}

            # Memory activity score (for sorting) - enhanced with new metrics
            rates["activity_score"] = (
                rates["minor_faults_s"] * 0.1
                + rates["major_faults_s"] * 10
                + rates["cmin_flt_s"] * 0.05
                + rates["cmaj_flt_s"] * 5
                + (rates["io_read_mb_s"] + rates["io_write_mb_s"]) * 100
                + abs(rates["rss_change_kb_s"]) * 0.01
                + proc.rss / 1024  # MB of RSS
            )

            process_rates.append((proc, rates))

        # Sort by activity score
        process_rates.sort(key=lambda x: x[1]["activity_score"], reverse=True)

        # Get top N processes
        top_processes = process_rates[: self.top_n]

        # Enrich top processes with expensive NUMA stats
        for proc, rates in top_processes:
            # If we merged processes, we can't easily get NUMA maps for the aggregate
            # But if it's a single process (not merged), or if merge mode is off:
            if proc.pid > 0:
                self._enrich_process_numa_stats(proc)

                # Re-calculate NUMA locality from newly acquired numa_maps data
                if proc.numa_total_pages > 0:
                    # Find which node has the most pages (primary node)
                    primary_node = (
                        max(
                            proc.numa_pages.keys(),
                            key=lambda n: proc.numa_pages.get(n, 0),
                        )
                        if proc.numa_pages
                        else 0
                    )
                    primary_pages = proc.numa_pages.get(primary_node, 0)
                    rates["numa_local_pct"] = (
                        primary_pages / proc.numa_total_pages
                    ) * 100
                    rates["numa_primary_node"] = primary_node

                    # Calculate distribution string (e.g., "N0:80% N1:20%")
                    numa_dist = []
                    for node_id in sorted(proc.numa_pages.keys()):
                        pct = (proc.numa_pages[node_id] / proc.numa_total_pages) * 100
                        if pct >= 1:  # Only show nodes with >= 1%
                            numa_dist.append(f"N{node_id}:{pct:.0f}%")
                    rates["numa_dist_str"] = " ".join(numa_dist) if numa_dist else "N/A"

                    # Memory in MB per node
                    rates["numa_mb"] = {
                        n: (p * PAGE_SIZE) / (1024 * 1024)
                        for n, p in proc.numa_pages.items()
                    }

        return top_processes

    def read_meminfo(self) -> Dict[str, int]:
        """Read /proc/meminfo"""
        meminfo = {}
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        key = parts[0].rstrip(":")
                        value = int(parts[1])
                        meminfo[key] = value
        except (IOError, ValueError):
            pass
        return meminfo

    def format_size(self, kb: int) -> str:
        """Format size in KB to human readable"""
        if kb >= 1024 * 1024:
            return f"{kb / (1024 * 1024):.1f} GB"
        elif kb >= 1024:
            return f"{kb / 1024:.1f} MB"
        else:
            return f"{kb} KB"

    def format_rate(self, rate: float, unit: str = "/s") -> str:
        """Format rate with appropriate suffix"""
        if rate >= 1000000:
            return f"{rate / 1000000:.1f}M{unit}"
        elif rate >= 1000:
            return f"{rate / 1000:.1f}K{unit}"
        else:
            return f"{rate:.1f}{unit}"

    def run_curses(self, stdscr):
        """Main curses-based display loop"""
        curses.curs_set(0)  # Hide cursor
        curses.use_default_colors()

        # Initialize color pairs
        curses.init_pair(1, curses.COLOR_GREEN, -1)
        curses.init_pair(2, curses.COLOR_YELLOW, -1)
        curses.init_pair(3, curses.COLOR_RED, -1)
        curses.init_pair(4, curses.COLOR_CYAN, -1)
        curses.init_pair(5, curses.COLOR_MAGENTA, -1)

        stdscr.nodelay(True)  # Non-blocking input

        prev_vmstats = self.read_vmstat()
        prev_numa_stats = self.read_numa_stats()
        initial_processes = self.get_process_mem_stats()
        # Key by name for merge mode, by pid otherwise
        if self.merge_by_name:
            prev_proc_stats = {
                p.name: p for p in self.merge_processes_by_name(initial_processes)
            }
        else:
            prev_proc_stats = {p.pid: p for p in initial_processes}
        prev_time = time.time()

        iteration = 0

        while True:
            try:
                # Check for quit
                key = stdscr.getch()
                if key == ord("q") or key == ord("Q"):
                    break

                # Wait for interval
                time.sleep(self.interval)

                # Collect current stats
                current_time = time.time()
                elapsed = current_time - prev_time

                current_vmstats = self.read_vmstat()
                current_numa_stats = self.read_numa_stats()
                current_processes = self.get_process_mem_stats()
                meminfo = self.read_meminfo()

                # Calculate rates
                rates = self.calculate_bandwidth_rates(
                    current_vmstats, prev_vmstats, elapsed
                )

                node_rates = self.calculate_node_bandwidth(
                    current_numa_stats, prev_numa_stats, elapsed
                )

                top_procs = self.get_top_memory_processes(
                    current_processes, prev_proc_stats, elapsed
                )

                # Get zone/PCP stats for lock contention analysis
                pcp_stats = self.read_pcp_stats()
                zone_lock_metrics = self.calculate_zone_lock_metrics(rates, pcp_stats)

                # Update trends
                if rates:
                    self.trends["mem_bandwidth"].add(rates.get("total_mb_s", 0))
                    self.trends["numa_local"].add(rates.get("numa_local_s", 0))
                    self.trends["numa_remote"].add(rates.get("numa_other_s", 0))
                    self.trends["page_faults"].add(rates.get("minor_faults_s", 0))
                    self.trends["major_faults"].add(rates.get("major_faults_s", 0))

                # Update zone lock contention trends
                if zone_lock_metrics:
                    self.trends["zone_lock_pressure"].add(
                        zone_lock_metrics.get("lock_contention_pressure", 0)
                    )
                    self.trends["pcp_refill_rate"].add(
                        zone_lock_metrics.get("zone_lock_acq_per_sec", 0)
                    )

                for node_id, nr in node_rates.items():
                    self.node_trends[node_id].add(nr.get("total_mb_s", 0))

                # Get node memory info
                node_meminfo = {
                    node_id: self.read_numa_node_meminfo(node_id)
                    for node_id in self.numa_nodes
                }

                # Display
                self._draw_screen(
                    stdscr,
                    rates,
                    node_rates,
                    node_meminfo,
                    meminfo,
                    top_procs,
                    iteration,
                    zone_lock_metrics,
                    pcp_stats,
                )

                # Update previous stats - key by name for merge mode, by pid otherwise
                prev_vmstats = current_vmstats
                prev_numa_stats = current_numa_stats
                if self.merge_by_name:
                    prev_proc_stats = {
                        p.name: p
                        for p in self.merge_processes_by_name(current_processes)
                    }
                else:
                    prev_proc_stats = {p.pid: p for p in current_processes}
                prev_time = current_time
                iteration += 1

            except KeyboardInterrupt:
                break

    def _draw_screen(
        self,
        stdscr,
        rates: Dict,
        node_rates: Dict,
        node_meminfo: Dict,
        meminfo: Dict,
        top_procs: List,
        iteration: int,
        zone_lock_metrics: Optional[Dict] = None,
        pcp_stats: Optional[Dict] = None,
    ):
        """Draw the monitoring screen"""
        stdscr.clear()
        height, width = stdscr.getmaxyx()

        row = 0

        # Header
        header = f" Memory Bandwidth Monitor - AMD EPYC - {self.cpu_count} CPUs, {len(self.numa_nodes)} NUMA Nodes "
        stdscr.addstr(row, 0, "=" * min(width - 1, 80))
        row += 1
        stdscr.addstr(row, 0, header[: width - 1], curses.A_BOLD | curses.color_pair(4))
        row += 1
        stdscr.addstr(
            row,
            0,
            f" Refresh: {self.interval}s | Press 'q' to quit | Iteration: {iteration}",
        )
        row += 1
        stdscr.addstr(row, 0, "=" * min(width - 1, 80))
        row += 2

        # System Memory Overview
        if meminfo:
            mem_total = meminfo.get("MemTotal", 0)
            mem_free = meminfo.get("MemFree", 0)
            mem_available = meminfo.get("MemAvailable", 0)
            cached = meminfo.get("Cached", 0)
            buffers = meminfo.get("Buffers", 0)

            stdscr.addstr(row, 0, "SYSTEM MEMORY", curses.A_BOLD | curses.color_pair(5))
            row += 1
            stdscr.addstr(
                row,
                0,
                f"  Total: {self.format_size(mem_total)}  "
                f"Available: {self.format_size(mem_available)}  "
                f"Free: {self.format_size(mem_free)}  "
                f"Cached: {self.format_size(cached + buffers)}",
            )
            row += 2

        # Memory Bandwidth Estimation
        if rates:
            bw_trend = self.trends["mem_bandwidth"].trend()
            stdscr.addstr(
                row,
                0,
                "MEMORY BANDWIDTH ESTIMATION",
                curses.A_BOLD | curses.color_pair(5),
            )
            row += 1

            # Color based on bandwidth
            total_bw = rates.get("total_mb_s", 0)
            if total_bw > 1000:
                color = curses.color_pair(3)  # Red for high
            elif total_bw > 100:
                color = curses.color_pair(2)  # Yellow for medium
            else:
                color = curses.color_pair(1)  # Green for low

            stdscr.addstr(
                row,
                0,
                f"  Page I/O: Read {rates.get('read_mb_s', 0):.1f} MB/s | "
                f"Write {rates.get('write_mb_s', 0):.1f} MB/s | "
                f"Total {total_bw:.1f} MB/s {bw_trend}",
                color,
            )
            row += 2

        # Per-NUMA Node Statistics
        stdscr.addstr(
            row, 0, "PER-NUMA NODE STATISTICS", curses.A_BOLD | curses.color_pair(5)
        )
        row += 1

        for node_id in self.numa_nodes:
            nr = node_rates.get(node_id, {})
            nm = node_meminfo.get(node_id, NodeMemInfo(node_id=node_id))
            trend = self.node_trends.get(node_id, TrendData()).trend()

            local_mb = nr.get("local_mb_s", 0)
            remote_mb = nr.get("remote_mb_s", 0)
            total_mb = nr.get("total_mb_s", 0)

            # Memory usage percentage
            if nm.mem_total > 0:
                used_pct = ((nm.mem_total - nm.mem_free) / nm.mem_total) * 100
            else:
                used_pct = 0

            # Color based on remote access (ideally should be low)
            if remote_mb > local_mb and remote_mb > 10:
                color = curses.color_pair(3)  # Red - too much remote
            elif remote_mb > 1:
                color = curses.color_pair(2)  # Yellow - some remote
            else:
                color = curses.color_pair(1)  # Green - mostly local

            stdscr.addstr(row, 0, f"  Node {node_id}: ", curses.A_BOLD)
            stdscr.addstr(
                f"Local {local_mb:.1f} MB/s | Remote {remote_mb:.1f} MB/s | "
                f"Total {total_mb:.1f} MB/s {trend}",
                color,
            )
            row += 1
            stdscr.addstr(
                row,
                0,
                f"          Mem: {self.format_size(nm.mem_total - nm.mem_free)} used / "
                f"{self.format_size(nm.mem_total)} ({used_pct:.1f}%)",
            )
            row += 1

        row += 1

        # NUMA Locality
        if rates:
            local_pct = rates.get("numa_local_pct", 100)
            local_trend = self.trends["numa_local"].trend()
            remote_trend = self.trends["numa_remote"].trend()

            if local_pct >= 90:
                color = curses.color_pair(1)  # Green
            elif local_pct >= 70:
                color = curses.color_pair(2)  # Yellow
            else:
                color = curses.color_pair(3)  # Red

            stdscr.addstr(row, 0, "NUMA LOCALITY", curses.A_BOLD | curses.color_pair(5))
            row += 1
            stdscr.addstr(
                row,
                0,
                f"  Local: {self.format_rate(rates.get('numa_local_s', 0))} {local_trend} | "
                f"Remote: {self.format_rate(rates.get('numa_other_s', 0))} {remote_trend} | "
                f"Locality: ",
                curses.A_NORMAL,
            )
            stdscr.addstr(f"{local_pct:.1f}%", color)
            row += 2

        # Page Faults and Memory Pressure
        if rates:
            fault_trend = self.trends["page_faults"].trend()
            major_trend = self.trends["major_faults"].trend()
            pressure = rates.get("memory_pressure", 0)

            if pressure > 50:
                pressure_color = curses.color_pair(3)
            elif pressure > 10:
                pressure_color = curses.color_pair(2)
            else:
                pressure_color = curses.color_pair(1)

            stdscr.addstr(
                row,
                0,
                "PAGE FAULTS & MEMORY PRESSURE",
                curses.A_BOLD | curses.color_pair(5),
            )
            row += 1
            stdscr.addstr(
                row,
                0,
                f"  Minor Faults: {self.format_rate(rates.get('minor_faults_s', 0))} {fault_trend} | "
                f"Major Faults: {self.format_rate(rates.get('major_faults_s', 0))} {major_trend}",
            )
            row += 1
            stdscr.addstr(
                row,
                0,
                f"  Page Alloc: {self.format_rate(rates.get('alloc_pages_s', 0))} | "
                f"Page Free: {self.format_rate(rates.get('free_pages_s', 0))} | "
                f"Swap: {self.format_rate(rates.get('swap_activity_s', 0))}",
            )
            row += 1
            stdscr.addstr(row, 0, "  Memory Pressure: ")
            stdscr.addstr(f"{pressure:.1f}/100", pressure_color)
            row += 2

        # Zone Lock Contention Metrics (critical for diagnosing allocator contention)
        if zone_lock_metrics and row < height - 15:
            lock_pressure = zone_lock_metrics.get("lock_contention_pressure", 0)
            lock_trend = self.trends["zone_lock_pressure"].trend()

            # Color based on contention pressure
            if lock_pressure > 50:
                lock_color = curses.color_pair(3)  # Red - high contention
            elif lock_pressure > 10:
                lock_color = curses.color_pair(2)  # Yellow - moderate
            else:
                lock_color = curses.color_pair(1)  # Green - low

            stdscr.addstr(
                row, 0, "ZONE LOCK & PCP METRICS", curses.A_BOLD | curses.color_pair(5)
            )
            row += 1

            # PCP configuration
            pcp_high = zone_lock_metrics.get("pcp_high", 0)
            pcp_batch = zone_lock_metrics.get("pcp_batch", 0)
            pcp_cached = zone_lock_metrics.get("pcp_total_cached", 0)
            stdscr.addstr(
                row,
                0,
                f"  PCP Config: high={pcp_high} batch={pcp_batch} | "
                f"Cached: {pcp_cached:,} pages",
            )
            row += 1

            # Lock acquisition rate
            lock_acq = zone_lock_metrics.get("zone_lock_acq_per_sec", 0)
            pcp_exhaust = zone_lock_metrics.get("pcp_exhaust_time_ms", 0)
            if pcp_exhaust == float("inf"):
                exhaust_str = "N/A"
            else:
                exhaust_str = f"{pcp_exhaust:.1f}ms"

            stdscr.addstr(
                row,
                0,
                f"  Zone Lock Acq: {self.format_rate(lock_acq)} | "
                f"PCP Exhaust Time: {exhaust_str}",
            )
            row += 1

            # Contention pressure indicator
            stdscr.addstr(row, 0, f"  Lock Contention Pressure: ")
            stdscr.addstr(f"{lock_pressure:.1f}/100 {lock_trend}", lock_color)

            # Add warning if high contention
            if lock_pressure > 30:
                stdscr.addstr(
                    " [HIGH - consider NUMA separation]", curses.color_pair(3)
                )
            row += 1

            # Allocation stalls (direct evidence of contention)
            allocstall = rates.get("allocstall_s", 0)
            compact_stall = rates.get("compact_stall_s", 0)
            if allocstall > 0 or compact_stall > 0:
                stdscr.addstr(
                    row,
                    0,
                    f"  Alloc Stalls: {self.format_rate(allocstall)} | "
                    f"Compact Stalls: {self.format_rate(compact_stall)}",
                    curses.color_pair(3) if allocstall > 0 else curses.A_NORMAL,
                )
                row += 1

            row += 1

        # Top Processes
        if self.merge_by_name:
            stdscr.addstr(
                row,
                0,
                "TOP PROCESSES BY MEMORY ACTIVITY (merged by name)",
                curses.A_BOLD | curses.color_pair(5),
            )
        else:
            stdscr.addstr(
                row,
                0,
                "TOP PROCESSES BY MEMORY ACTIVITY",
                curses.A_BOLD | curses.color_pair(5),
            )
        row += 1

        # Header - include Count column for merged mode
        # Enhanced header with syscalls, allocation rate, and NUMA distribution
        if self.merge_by_name:
            header_fmt = f"  {'#':>4} {'Name':<12} {'RSS':>7} {'MinF/s':>7} {'Alloc':>7} {'Sys/s':>7} {'CPU%':>5} {'NUMA Distribution':<20}"
        else:
            header_fmt = f"  {'PID':>7} {'Name':<12} {'RSS':>7} {'MinF/s':>7} {'Alloc':>7} {'Sys/s':>7} {'CPU%':>5} {'NUMA Distribution':<20}"
        stdscr.addstr(row, 0, header_fmt[: width - 1], curses.A_UNDERLINE)
        row += 1

        for proc, proc_rates in top_procs:
            if row >= height - 2:
                break

            # Highlight based on NUMA locality - red if < 70% local, yellow if < 90%
            numa_local_pct = proc_rates.get("numa_local_pct", 100)
            if proc_rates.get("major_faults_s", 0) > 10:
                color = curses.color_pair(3)  # Red - major faults indicate disk I/O
            elif numa_local_pct > 0 and numa_local_pct < 70:
                color = curses.color_pair(3)  # Red - poor NUMA locality
            elif proc_rates.get("alloc_mb_s", 0) > 100:
                color = curses.color_pair(3)  # Red - very high allocation rate
            elif numa_local_pct > 0 and numa_local_pct < 90:
                color = curses.color_pair(2)  # Yellow - moderate NUMA locality
            elif proc_rates.get("alloc_mb_s", 0) > 10:
                color = curses.color_pair(2)  # Yellow - high allocation rate
            elif proc_rates.get("activity_score", 0) > 1000:
                color = curses.color_pair(2)
            else:
                color = curses.A_NORMAL

            # Format allocation rate
            alloc_rate = proc_rates.get("alloc_mb_s", 0)
            if alloc_rate >= 1000:
                alloc_str = f"{alloc_rate / 1000:.1f}G/s"
            elif alloc_rate >= 1:
                alloc_str = f"{alloc_rate:.1f}M/s"
            else:
                alloc_str = f"{alloc_rate * 1024:.0f}K/s"

            # Format syscall rate
            syscall_rate = proc_rates.get("syscall_total_s", 0)
            if syscall_rate >= 1000000:
                syscall_str = f"{syscall_rate / 1000000:.1f}M"
            elif syscall_rate >= 1000:
                syscall_str = f"{syscall_rate / 1000:.1f}K"
            else:
                syscall_str = f"{syscall_rate:.0f}"

            # NUMA distribution string
            numa_str = proc_rates.get("numa_dist_str", "N/A")

            if self.merge_by_name:
                # Show count instead of PID for merged processes
                proc_line = (
                    f"  {proc.count:>4} {proc.name[:12]:<12} "
                    f"{self.format_size(proc.rss):>7} "
                    f"{proc_rates.get('minor_faults_s', 0):>7.0f} "
                    f"{alloc_str:>7} "
                    f"{syscall_str:>7} "
                    f"{proc_rates.get('cpu_total_pct', 0):>5.1f} "
                    f"{numa_str:<20}"
                )
            else:
                proc_line = (
                    f"  {proc.pid:>7} {proc.name[:12]:<12} "
                    f"{self.format_size(proc.rss):>7} "
                    f"{proc_rates.get('minor_faults_s', 0):>7.0f} "
                    f"{alloc_str:>7} "
                    f"{syscall_str:>7} "
                    f"{proc_rates.get('cpu_total_pct', 0):>5.1f} "
                    f"{numa_str:<20}"
                )

            stdscr.addstr(row, 0, proc_line[: width - 1], color)
            row += 1

        # Footer
        stdscr.addstr(
            height - 1,
            0,
            " Legend: ↑ increasing | ↓ decreasing | → stable | ? insufficient data"[
                : width - 1
            ],
        )

        stdscr.refresh()

    def run_simple(self):
        """Simple non-curses display mode (fallback)"""
        prev_vmstats = self.read_vmstat()
        prev_numa_stats = self.read_numa_stats()
        initial_processes = self.get_process_mem_stats()
        # Key by name for merge mode, by pid otherwise
        if self.merge_by_name:
            prev_proc_stats = {
                p.name: p for p in self.merge_processes_by_name(initial_processes)
            }
        else:
            prev_proc_stats = {p.pid: p for p in initial_processes}
        prev_time = time.time()

        iteration = 0

        print("\nMemory Bandwidth Monitor - AMD EPYC")
        print(f"CPUs: {self.cpu_count}, NUMA Nodes: {len(self.numa_nodes)}")
        print(f"Refresh interval: {self.interval}s")
        if self.merge_by_name:
            print("Process merging: enabled (processes with same name are merged)")
        print("Press Ctrl+C to exit\n")

        try:
            while True:
                time.sleep(self.interval)

                current_time = time.time()
                elapsed = current_time - prev_time

                current_vmstats = self.read_vmstat()
                current_numa_stats = self.read_numa_stats()
                current_processes = self.get_process_mem_stats()
                meminfo = self.read_meminfo()

                rates = self.calculate_bandwidth_rates(
                    current_vmstats, prev_vmstats, elapsed
                )

                node_rates = self.calculate_node_bandwidth(
                    current_numa_stats, prev_numa_stats, elapsed
                )

                top_procs = self.get_top_memory_processes(
                    current_processes, prev_proc_stats, elapsed
                )

                # Update trends
                if rates:
                    self.trends["mem_bandwidth"].add(rates.get("total_mb_s", 0))
                    self.trends["numa_local"].add(rates.get("numa_local_s", 0))
                    self.trends["numa_remote"].add(rates.get("numa_other_s", 0))

                # Clear screen
                print("\033[2J\033[H", end="")

                # Print stats
                print(f"{'=' * 70}")
                print(f" Memory Bandwidth Monitor - Iteration {iteration}")
                print(f"{'=' * 70}\n")

                # Memory
                mem_total = meminfo.get("MemTotal", 0)
                mem_available = meminfo.get("MemAvailable", 0)
                print(
                    f"SYSTEM MEMORY: {self.format_size(mem_available)} available / "
                    f"{self.format_size(mem_total)} total\n"
                )

                # Bandwidth
                if rates:
                    bw_trend = self.trends["mem_bandwidth"].trend()
                    print(
                        f"BANDWIDTH: Read {rates.get('read_mb_s', 0):.1f} MB/s | "
                        f"Write {rates.get('write_mb_s', 0):.1f} MB/s | "
                        f"Total {rates.get('total_mb_s', 0):.1f} MB/s {bw_trend}\n"
                    )

                # NUMA nodes
                print("NUMA NODES:")
                for node_id in self.numa_nodes:
                    nr = node_rates.get(node_id, {})
                    nm = self.read_numa_node_meminfo(node_id)
                    print(
                        f"  Node {node_id}: Local {nr.get('local_mb_s', 0):.1f} MB/s | "
                        f"Remote {nr.get('remote_mb_s', 0):.1f} MB/s | "
                        f"Mem {self.format_size(nm.mem_total - nm.mem_free)} used"
                    )
                print()

                # NUMA locality
                if rates:
                    print(
                        f"NUMA LOCALITY: {rates.get('numa_local_pct', 100):.1f}% local accesses\n"
                    )

                # Page faults
                if rates:
                    print(
                        f"PAGE FAULTS: Minor {self.format_rate(rates.get('minor_faults_s', 0))} | "
                        f"Major {self.format_rate(rates.get('major_faults_s', 0))} | "
                        f"Pressure {rates.get('memory_pressure', 0):.1f}/100\n"
                    )

                # Zone Lock / PCP Metrics
                pcp_stats = self.read_pcp_stats()
                zone_lock_metrics = self.calculate_zone_lock_metrics(rates, pcp_stats)

                if zone_lock_metrics:
                    print("ZONE LOCK & PCP METRICS:")
                    pcp_high = zone_lock_metrics.get("pcp_high", 0)
                    pcp_batch = zone_lock_metrics.get("pcp_batch", 0)
                    pcp_cached = zone_lock_metrics.get("pcp_total_cached", 0)
                    print(
                        f"  PCP Config: high={pcp_high} batch={pcp_batch} | Cached: {pcp_cached:,} pages"
                    )

                    lock_acq = zone_lock_metrics.get("zone_lock_acq_per_sec", 0)
                    pcp_exhaust = zone_lock_metrics.get("pcp_exhaust_time_ms", 0)
                    exhaust_str = (
                        f"{pcp_exhaust:.1f}ms" if pcp_exhaust != float("inf") else "N/A"
                    )
                    print(
                        f"  Zone Lock Acq: {self.format_rate(lock_acq)} | PCP Exhaust Time: {exhaust_str}"
                    )

                    lock_pressure = zone_lock_metrics.get("lock_contention_pressure", 0)
                    lock_trend = self.trends["zone_lock_pressure"].trend()
                    pressure_indicator = ""
                    if lock_pressure > 50:
                        pressure_indicator = " [HIGH - consider NUMA separation]"
                    elif lock_pressure > 30:
                        pressure_indicator = " [MODERATE]"
                    print(
                        f"  Lock Contention Pressure: {lock_pressure:.1f}/100 {lock_trend}{pressure_indicator}"
                    )

                    allocstall = rates.get("allocstall_s", 0)
                    if allocstall > 0:
                        print(
                            f"  Alloc Stalls: {self.format_rate(allocstall)} [CONTENTION DETECTED]"
                        )
                    print()

                # Top processes
                if self.merge_by_name:
                    print("TOP PROCESSES (merged by name):")
                    print(
                        f"  {'#':>4} {'Name':<12} {'RSS':>7} {'MinF/s':>7} {'Alloc':>7} {'Sys/s':>7} {'CPU%':>5} {'NUMA Distribution':<20}"
                    )
                    print(
                        f"  {'-' * 4} {'-' * 12} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 5} {'-' * 20}"
                    )
                else:
                    print("TOP PROCESSES:")
                    print(
                        f"  {'PID':>7} {'Name':<12} {'RSS':>7} {'MinF/s':>7} {'Alloc':>7} {'Sys/s':>7} {'CPU%':>5} {'NUMA Distribution':<20}"
                    )
                    print(
                        f"  {'-' * 7} {'-' * 12} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 5} {'-' * 20}"
                    )

                for proc, proc_rates in top_procs[:10]:
                    # Format allocation rate
                    alloc_rate = proc_rates.get("alloc_mb_s", 0)
                    if alloc_rate >= 1000:
                        alloc_str = f"{alloc_rate / 1000:.1f}G"
                    elif alloc_rate >= 1:
                        alloc_str = f"{alloc_rate:.1f}M"
                    else:
                        alloc_str = f"{alloc_rate * 1024:.0f}K"

                    # Format syscall rate
                    syscall_rate = proc_rates.get("syscall_total_s", 0)
                    if syscall_rate >= 1000000:
                        syscall_str = f"{syscall_rate / 1000000:.1f}M"
                    elif syscall_rate >= 1000:
                        syscall_str = f"{syscall_rate / 1000:.1f}K"
                    else:
                        syscall_str = f"{syscall_rate:.0f}"

                    # NUMA distribution
                    numa_str = proc_rates.get("numa_dist_str", "N/A")

                    if self.merge_by_name:
                        print(
                            f"  {proc.count:>4} {proc.name[:12]:<12} "
                            f"{self.format_size(proc.rss):>7} "
                            f"{proc_rates.get('minor_faults_s', 0):>7.0f} "
                            f"{alloc_str:>7} "
                            f"{syscall_str:>7} "
                            f"{proc_rates.get('cpu_total_pct', 0):>5.1f} "
                            f"{numa_str:<20}"
                        )
                    else:
                        print(
                            f"  {proc.pid:>7} {proc.name[:12]:<12} "
                            f"{self.format_size(proc.rss):>7} "
                            f"{proc_rates.get('minor_faults_s', 0):>7.0f} "
                            f"{alloc_str:>7} "
                            f"{syscall_str:>7} "
                            f"{proc_rates.get('cpu_total_pct', 0):>5.1f} "
                            f"{numa_str:<20}"
                        )

                print(f"\n{'=' * 70}")
                print("Legend: ↑ increasing | ↓ decreasing | → stable")

                prev_vmstats = current_vmstats
                prev_numa_stats = current_numa_stats
                # Key by name for merge mode, by pid otherwise
                if self.merge_by_name:
                    prev_proc_stats = {
                        p.name: p
                        for p in self.merge_processes_by_name(current_processes)
                    }
                else:
                    prev_proc_stats = {p.pid: p for p in current_processes}
                prev_time = current_time
                iteration += 1

        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")

    def run(self, use_curses: bool = True):
        """Start the monitor"""
        if use_curses:
            try:
                curses.wrapper(self.run_curses)
            except curses.error:
                print("Curses not available, falling back to simple mode")
                self.run_simple()
        else:
            self.run_simple()


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    print("\n\nReceived interrupt signal. Exiting...")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Monitor memory bandwidth and cache statistics on AMD EPYC systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Run with default 1 second interval
  %(prog)s --interval 2       # Run with 2 second refresh interval
  %(prog)s --top 20           # Show top 20 processes
  %(prog)s --simple           # Use simple text output (no curses)
  %(prog)s --merge            # Merge processes with same name
  %(prog)s -m -n 15           # Merge processes and show top 15
  
Notes:
  - Run without root for basic monitoring
  - Some metrics may require root access for full detail
  - Designed for AMD EPYC Milan systems with 2 NUMA nodes
  - Use --merge to aggregate stats for multi-process applications

Zone Lock & PCP Metrics (for diagnosing page allocator contention):
  - PCP high/batch: Per-CPU pagelist watermarks from /proc/zoneinfo
  - Zone Lock Acq/s: Estimated zone->lock acquisitions (alloc_rate / batch)
  - PCP Exhaust Time: How quickly each CPU depletes its page cache
  - Lock Contention Pressure: 0-100 indicator of likely lock contention
    - >50: HIGH - significant contention, consider NUMA separation
    - 30-50: MODERATE - noticeable contention
    - <30: LOW - minimal contention

Per-Process Metrics:
  - RSS: Resident Set Size (physical memory used)
  - MinF/s: Minor page faults per second (new page allocations)
  - Alloc: Estimated memory allocation rate (based on minor faults)
  - Sys/s: Total syscalls per second (read + write syscalls)
  - CPU: CPU utilization percentage (user + system time)
  - NUMA Distribution: Memory distribution across NUMA nodes (e.g., N0:80 N1:20)
    - Processes with <70 local memory are highlighted in red (poor locality)
    - Processes with <90 local memory are highlighted in yellow
        """,
    )

    parser.add_argument(
        "-i",
        "--interval",
        type=float,
        default=1.0,
        help="Refresh interval in seconds (default: 1.0)",
    )

    parser.add_argument(
        "-n",
        "--top",
        type=int,
        default=10,
        help="Number of top processes to show (default: 10)",
    )

    parser.add_argument(
        "-s",
        "--simple",
        action="store_true",
        help="Use simple text output instead of curses",
    )

    parser.add_argument(
        "-m",
        "--merge",
        action="store_true",
        help="Merge processes with the same name (aggregate stats)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.interval < 0.1:
        print("Warning: Interval too small, setting to 0.1 seconds")
        args.interval = 0.1
    elif args.interval > 60:
        print("Warning: Interval very large, setting to 60 seconds")
        args.interval = 60

    if args.top < 1:
        args.top = 1
    elif args.top > 100:
        args.top = 100

    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Check if we're running as root
    is_root = os.geteuid() == 0
    if not is_root:
        print("Note: Running without root. Some metrics may be limited.")
        print("      Run with sudo for full perf support.\n")

    # Check required files exist
    required_files = ["/proc/vmstat", "/proc/meminfo"]
    for f in required_files:
        if not os.path.exists(f):
            print(f"Error: Required file {f} not found. Are you running on Linux?")
            sys.exit(1)

    # Start monitoring
    monitor = MemoryBandwidthMonitor(
        interval=args.interval, top_n=args.top, merge_by_name=args.merge
    )

    monitor.run(use_curses=not args.simple)


if __name__ == "__main__":
    main()
