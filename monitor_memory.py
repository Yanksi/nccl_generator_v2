#!/usr/bin/env python3
"""
Memory monitoring script for a process and all its children.

Usage:
    python monitor_memory.py <command>
    python monitor_memory.py "uv run main_v2.py --trace_dir ... --dask"
    python monitor_memory.py --pid <pid>  # Monitor existing process

Examples:
    python monitor_memory.py "sleep 10"
    python monitor_memory.py --pid 12345
"""

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import time
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any

try:
    import psutil
except ImportError:
    print("Error: psutil is required. Install with: pip install psutil")
    sys.exit(1)


# Cache directory for storing monitoring results
CACHE_DIR = pathlib.Path.home() / ".cache" / "monitor_memory"


def get_cache_key(command: str) -> str:
    """Generate a cache key from the command string."""
    return hashlib.sha256(command.encode()).hexdigest()[:16]


def get_cache_path(command: str) -> pathlib.Path:
    """Get the cache file path for a command."""
    return CACHE_DIR / f"{get_cache_key(command)}.json"


def save_cache(command: str, data: Dict[str, Any]) -> pathlib.Path:
    """Save monitoring results to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = get_cache_path(command)
    data['command'] = command
    data['cached_at'] = datetime.now().isoformat()
    with open(cache_path, 'w') as f:
        json.dump(data, f, indent=2)
    return cache_path


def load_cache(command: str) -> Optional[Dict[str, Any]]:
    """Load cached results for a command, if available."""
    cache_path = get_cache_path(command)
    if cache_path.exists():
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    return None


def clear_cache(command: str = None) -> int:
    """Clear cache for a specific command or all cache."""
    if command:
        cache_path = get_cache_path(command)
        if cache_path.exists():
            cache_path.unlink()
            return 1
        return 0
    else:
        # Clear all cache
        count = 0
        if CACHE_DIR.exists():
            for f in CACHE_DIR.glob("*.json"):
                f.unlink()
                count += 1
        return count


def print_cached_results(data: Dict[str, Any], show_details: bool = False,
                         group_threshold: float = 0.1):
    """Print cached monitoring results."""
    print(f"\n{'='*60}")
    print(f"MEMORY SUMMARY (cached)")
    print(f"{'='*60}")
    print(f"  Command:         {data.get('command', 'N/A')}")
    print(f"  Cached at:       {data.get('cached_at', 'N/A')}")
    print(f"  Exit code:       {data.get('exit_code', 'N/A')}")
    print(f"  Total time:      {data.get('elapsed', 0):.1f} seconds")
    print(f"  Peak RSS:        {format_memory(data.get('peak_rss', 0))}")
    if data.get('peak_uss', 0) > 0:
        print(f"  Peak USS:        {format_memory(data.get('peak_uss', 0))}")
        print(f"    - Parent:      {format_memory(data.get('peak_parent_uss', 0))}")
        print(f"    - Children:    {format_memory(data.get('peak_children_uss', 0))}")
        print(f"  ")
        print(f"  Note: USS (Unique Set Size) is memory unique to each process.")
        print(f"        Low children USS vs RSS indicates good copy-on-write sharing.")
    print(f"  Peak processes:  {data.get('peak_processes', 0)}")
    print(f"  Samples taken:   {data.get('num_samples', 0)}")
    
    peak_details = data.get('peak_details', [])
    if show_details and peak_details:
        print(f"\n  {'='*60}")
        print(f"  Process breakdown at peak USS:")
        print(f"  {'='*60}")
        # Group workers spawned together, then sort by USS descending
        grouped = group_processes_by_spawn_time(peak_details, group_threshold)
        sorted_details = sorted(grouped, key=lambda x: x['uss_mb'], reverse=True)[:15]
        print(f"  {'PID/Group':>12}  {'Role':<8}  {'Name':<14}  {'RSS':>12}  {'USS':>12}")
        print(f"  {'-'*12}  {'-'*8}  {'-'*14}  {'-'*12}  {'-'*12}")
        for d in sorted_details:
            pid_str = str(d['pid']) if isinstance(d['pid'], int) else d['pid']
            # Determine role
            if d.get('is_parent'):
                role = "launcher"
            elif d.get('is_main_script'):
                role = "main"
            elif d.get('is_group'):
                role = "workers"
            else:
                role = "worker"
            
            if d.get('is_group'):
                print(f"  {pid_str:>12}  {role:<8}  {d['name']:<14}  {format_memory(d['rss_mb']):>12}  {format_memory(d['uss_mb']):>12}")
                print(f"  {'':>12}  {'':>8}  {'(per worker)':<14}  {format_memory(d['avg_rss_mb']):>12}  {format_memory(d['avg_uss_mb']):>12}")
            else:
                print(f"  {pid_str:>12}  {role:<8}  {d['name']:<14}  {format_memory(d['rss_mb']):>12}  {format_memory(d['uss_mb']):>12}")
    
    print(f"{'='*60}\n")


def get_memory_info(pid: int) -> Tuple[float, float, float, float, float, int, List[dict]]:
    """
    Get memory info for a process and all its children.
    
    Returns:
        (total_rss_mb, total_uss_mb, total_pss_mb, parent_uss_mb, children_uss_mb, num_processes, process_details)
    """
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return 0, 0, 0, 0, 0, 0, []
    
    # Build process tree with depth information
    # depth 0 = root (monitored process, e.g., uv)
    # depth 1 = direct children (e.g., main Python script)
    # depth 2+ = grandchildren (e.g., worker processes)
    def get_descendants_with_depth(proc, depth=0):
        """Recursively get all descendants with their depth."""
        result = [(proc, depth)]
        try:
            for child in proc.children(recursive=False):
                result.extend(get_descendants_with_depth(child, depth + 1))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        return result
    
    processes_with_depth = get_descendants_with_depth(parent, 0)
    
    total_rss = 0
    total_uss = 0
    total_pss = 0
    parent_uss = 0
    children_uss = 0
    details = []
    
    for proc, depth in processes_with_depth:
        try:
            mem_info = proc.memory_info()
            rss = mem_info.rss
            total_rss += rss
            
            # Try to get USS/PSS (may not be available on all systems)
            try:
                mem_full = proc.memory_full_info()
                uss = mem_full.uss
                pss = getattr(mem_full, 'pss', 0)
                total_uss += uss
                total_pss += pss
                
                if depth == 0:  # Root process (e.g., uv launcher)
                    parent_uss = uss
                else:  # All descendants
                    children_uss += uss
            except (psutil.AccessDenied, AttributeError):
                uss = 0
                pss = 0
            
            details.append({
                'pid': proc.pid,
                'name': proc.name()[:20],
                'rss_mb': rss / (1024 * 1024),
                'uss_mb': uss / (1024 * 1024),
                'is_parent': depth == 0,
                'depth': depth,
                'create_time': proc.create_time(),
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    return (
        total_rss / (1024 * 1024),
        total_uss / (1024 * 1024),
        total_pss / (1024 * 1024),
        parent_uss / (1024 * 1024),
        children_uss / (1024 * 1024),
        len(details),
        details
    )


def format_memory(mb: float) -> str:
    """Format memory size with appropriate unit."""
    if mb >= 1024:
        return f"{mb/1024:.2f} GB"
    return f"{mb:.1f} MB"


def group_processes_by_spawn_time(details: List[dict], time_threshold: float = 0.1) -> List[dict]:
    """
    Group child processes that were spawned within a time threshold.
    Only groups processes at the same depth level (workers at depth 2+).
    Processes at depth 0 (root) and depth 1 (main script) are never grouped.
    
    Args:
        details: List of process detail dicts with 'create_time' and 'depth' fields
        time_threshold: Max seconds between spawn times to be considered same group.
                       Default 0.1s is tight enough to separate sequential Pool() calls
                       while grouping workers spawned by the same Pool.
    
    Returns:
        List of grouped process summaries
    """
    if not details:
        return []
    
    result = []
    
    # Separate by depth
    # depth 0: root process (uv launcher) - never grouped
    # depth 1: main script(s) - never grouped  
    # depth 2+: workers - can be grouped by spawn time
    ungrouped = []  # depth 0 and 1
    workers = []    # depth 2+
    
    for d in details:
        depth = d.get('depth', 0)
        if depth <= 1:
            # Mark depth 1 processes specially
            if depth == 1:
                d = d.copy()
                d['is_main_script'] = True
            ungrouped.append(d)
        else:
            workers.append(d)
    
    # Add ungrouped processes first
    result.extend(ungrouped)
    
    if not workers:
        return result
    
    # Sort workers by creation time
    workers_sorted = sorted(workers, key=lambda x: x.get('create_time', 0))
    
    # Group by spawn time
    groups = []
    current_group = [workers_sorted[0]]
    
    for worker in workers_sorted[1:]:
        # Check if this worker was spawned close to the last one in current group
        time_diff = worker.get('create_time', 0) - current_group[-1].get('create_time', 0)
        if time_diff <= time_threshold:
            current_group.append(worker)
        else:
            groups.append(current_group)
            current_group = [worker]
    
    if current_group:
        groups.append(current_group)
    
    # Aggregate each group
    for group in groups:
        if len(group) == 1:
            # Single process, keep as-is
            result.append(group[0])
        else:
            # Multiple processes - aggregate
            total_rss = sum(p['rss_mb'] for p in group)
            total_uss = sum(p['uss_mb'] for p in group)
            avg_rss = total_rss / len(group)
            avg_uss = total_uss / len(group)
            
            # Use first process name as representative
            name = group[0]['name']
            
            result.append({
                'pid': f"{len(group)} workers",
                'name': name,
                'rss_mb': total_rss,
                'uss_mb': total_uss,
                'avg_rss_mb': avg_rss,
                'avg_uss_mb': avg_uss,
                'is_parent': False,
                'is_group': True,
                'count': len(group),
            })
    
    return result


def clear_line():
    """Clear current line in terminal."""
    print("\033[2K\033[G", end="")


def move_up(n: int):
    """Move cursor up n lines."""
    if n > 0:
        print(f"\033[{n}A", end="")


def monitor_process(pid: int, interval: float = 0.5, show_details: bool = False, 
                    group_threshold: float = 0.1):
    """Monitor memory usage of a process and its children."""
    peak_rss = 0
    peak_uss = 0
    peak_parent_uss = 0
    peak_children_uss = 0
    peak_processes = 0
    start_time = time.time()
    last_lines = 0
    
    print(f"\n{'='*60}")
    print(f"Monitoring PID {pid} and children (Ctrl+C to stop)")
    print(f"{'='*60}\n")
    
    try:
        while True:
            rss, uss, pss, parent_uss, children_uss, num_procs, details = get_memory_info(pid)
            
            if num_procs == 0:
                print("\nProcess ended.")
                break
            
            # Update peaks
            peak_rss = max(peak_rss, rss)
            peak_uss = max(peak_uss, uss)
            peak_parent_uss = max(peak_parent_uss, parent_uss)
            peak_children_uss = max(peak_children_uss, children_uss)
            peak_processes = max(peak_processes, num_procs)
            
            elapsed = time.time() - start_time
            
            # Move cursor up to overwrite previous output
            move_up(last_lines)
            
            # Build output
            lines = []
            lines.append(f"Time: {elapsed:.1f}s | Processes: {num_procs}")
            lines.append(f"")
            lines.append(f"  Current RSS:  {format_memory(rss):>12}  (Peak: {format_memory(peak_rss)})")
            if uss > 0:
                lines.append(f"  Current USS:  {format_memory(uss):>12}  (Peak: {format_memory(peak_uss)})")
                lines.append(f"    - Parent:   {format_memory(parent_uss):>12}  (Peak: {format_memory(peak_parent_uss)})")
                lines.append(f"    - Children: {format_memory(children_uss):>12}  (Peak: {format_memory(peak_children_uss)})")
                lines.append(f"  Current PSS:  {format_memory(pss):>12}")
            lines.append(f"")
            
            if show_details and details:
                # Group workers spawned together, then sort by USS descending
                grouped = group_processes_by_spawn_time(details, group_threshold)
                sorted_details = sorted(grouped, key=lambda x: x['uss_mb'], reverse=True)[:10]
                lines.append(f"  Top processes by memory:")
                lines.append(f"  {'PID/Group':>12}  {'Role':<8}  {'Name':<14}  {'RSS':>10}  {'USS':>10}")
                lines.append(f"  {'-'*12}  {'-'*8}  {'-'*14}  {'-'*10}  {'-'*10}")
                for d in sorted_details:
                    pid_str = str(d['pid']) if isinstance(d['pid'], int) else d['pid']
                    # Determine role
                    if d.get('is_parent'):
                        role = "launcher"
                    elif d.get('is_main_script'):
                        role = "main"
                    elif d.get('is_group'):
                        role = "workers"
                    else:
                        role = "worker"
                    
                    if d.get('is_group'):
                        lines.append(f"  {pid_str:>12}  {role:<8}  {d['name']:<14}  {format_memory(d['rss_mb']):>10}  {format_memory(d['uss_mb']):>10}")
                        lines.append(f"  {'':>12}  {'':>8}  {'(per worker)':<14}  {format_memory(d['avg_rss_mb']):>10}  {format_memory(d['avg_uss_mb']):>10}")
                    else:
                        lines.append(f"  {pid_str:>12}  {role:<8}  {d['name']:<14}  {format_memory(d['rss_mb']):>10}  {format_memory(d['uss_mb']):>10}")
                lines.append("")
            
            # Print lines
            for line in lines:
                clear_line()
                print(line)
            
            last_lines = len(lines)
            time.sleep(interval)
            
    except KeyboardInterrupt:
        pass
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Total time:      {time.time() - start_time:.1f} seconds")
    print(f"  Peak RSS:        {format_memory(peak_rss)}")
    if peak_uss > 0:
        print(f"  Peak USS:        {format_memory(peak_uss)}")
        print(f"    - Parent:      {format_memory(peak_parent_uss)}")
        print(f"    - Children:    {format_memory(peak_children_uss)}")
    print(f"  Peak processes:  {peak_processes}")
    print(f"{'='*60}\n")


def run_and_monitor(command: str, interval: float = 0.5, show_details: bool = False,
                    group_threshold: float = 0.1, save_to_cache: bool = True,
                    force_cache: bool = False):
    """Run a command and monitor its memory usage."""
    print(f"Starting command: {command}")
    
    # Start the process
    proc = subprocess.Popen(
        command,
        shell=True,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    
    # Give it a moment to start
    time.sleep(0.2)
    
    # Monitor in a separate flow
    peak_rss = 0
    peak_uss = 0
    peak_parent_uss = 0
    peak_children_uss = 0
    peak_processes = 0
    start_time = time.time()
    samples = []
    peak_details = []  # Store details at peak USS
    
    print(f"\n{'='*60}")
    print(f"Monitoring PID {proc.pid} (command running in foreground)")
    print(f"Memory stats will be shown after command completes")
    print(f"{'='*60}\n")
    
    try:
        while proc.poll() is None:
            rss, uss, pss, parent_uss, children_uss, num_procs, details = get_memory_info(proc.pid)
            
            if num_procs > 0:
                peak_rss = max(peak_rss, rss)
                if uss > peak_uss:
                    peak_uss = uss
                    peak_details = details  # Capture details at peak USS
                peak_parent_uss = max(peak_parent_uss, parent_uss)
                peak_children_uss = max(peak_children_uss, children_uss)
                peak_processes = max(peak_processes, num_procs)
                samples.append((time.time() - start_time, rss, uss, parent_uss, children_uss, num_procs))
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()
    
    # Print summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"MEMORY SUMMARY")
    print(f"{'='*60}")
    print(f"  Exit code:       {proc.returncode}")
    print(f"  Total time:      {elapsed:.1f} seconds")
    print(f"  Peak RSS:        {format_memory(peak_rss)}")
    if peak_uss > 0:
        print(f"  Peak USS:        {format_memory(peak_uss)}")
        print(f"    - Parent:      {format_memory(peak_parent_uss)}")
        print(f"    - Children:    {format_memory(peak_children_uss)}")
        print(f"  ")
        print(f"  Note: USS (Unique Set Size) is memory unique to each process.")
        print(f"        Low children USS vs RSS indicates good copy-on-write sharing.")
    print(f"  Peak processes:  {peak_processes}")
    print(f"  Samples taken:   {len(samples)}")
    
    if show_details and peak_details:
        print(f"\n  {'='*60}")
        print(f"  Process breakdown at peak USS:")
        print(f"  {'='*60}")
        # Group workers spawned together, then sort by USS descending
        grouped = group_processes_by_spawn_time(peak_details, group_threshold)
        sorted_details = sorted(grouped, key=lambda x: x['uss_mb'], reverse=True)[:15]
        print(f"  {'PID/Group':>12}  {'Role':<8}  {'Name':<14}  {'RSS':>12}  {'USS':>12}")
        print(f"  {'-'*12}  {'-'*8}  {'-'*14}  {'-'*12}  {'-'*12}")
        for d in sorted_details:
            pid_str = str(d['pid']) if isinstance(d['pid'], int) else d['pid']
            # Determine role
            if d.get('is_parent'):
                role = "launcher"
            elif d.get('is_main_script'):
                role = "main"
            elif d.get('is_group'):
                role = "workers"
            else:
                role = "worker"
            
            if d.get('is_group'):
                print(f"  {pid_str:>12}  {role:<8}  {d['name']:<14}  {format_memory(d['rss_mb']):>12}  {format_memory(d['uss_mb']):>12}")
                print(f"  {'':>12}  {'':>8}  {'(per worker)':<14}  {format_memory(d['avg_rss_mb']):>12}  {format_memory(d['avg_uss_mb']):>12}")
            else:
                print(f"  {pid_str:>12}  {role:<8}  {d['name']:<14}  {format_memory(d['rss_mb']):>12}  {format_memory(d['uss_mb']):>12}")
    
    print(f"{'='*60}\n")
    
    # Save to cache if requested and appropriate
    # By default, only cache successful runs (exit code 0)
    # Use force_cache=True to cache even on failure
    should_cache = save_to_cache and (proc.returncode == 0 or force_cache)
    if should_cache:
        cache_data = {
            'exit_code': proc.returncode,
            'elapsed': elapsed,
            'peak_rss': peak_rss,
            'peak_uss': peak_uss,
            'peak_parent_uss': peak_parent_uss,
            'peak_children_uss': peak_children_uss,
            'peak_processes': peak_processes,
            'num_samples': len(samples),
            'peak_details': peak_details,
        }
        cache_path = save_cache(command, cache_data)
        print(f"Results cached to: {cache_path}")
    elif save_to_cache and proc.returncode != 0:
        print(f"Results not cached (exit code {proc.returncode}). Use --force-cache to cache failed runs.")
    
    return proc.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Monitor memory usage of a process and its children",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "uv run main_v2.py --dask"     # Run and monitor command
  %(prog)s --pid 12345                     # Monitor existing process
  %(prog)s --pid 12345 --details           # Show per-process breakdown
  %(prog)s --use-cached "uv run ..."       # Show cached results for command
  %(prog)s --clear-cache                   # Clear all cached results
        """
    )
    parser.add_argument("command", nargs="?", help="Command to run and monitor")
    parser.add_argument("--pid", type=int, help="PID of existing process to monitor")
    parser.add_argument("--interval", "-i", type=float, default=0.5, 
                        help="Sampling interval in seconds (default: 0.5)")
    parser.add_argument("--details", "-d", action="store_true",
                        help="Show per-process memory breakdown")
    parser.add_argument("--group-threshold", "-g", type=float, default=0.1,
                        help="Time threshold in seconds for grouping workers spawned together (default: 0.1)")
    parser.add_argument("--use-cached", "-c", action="store_true",
                        help="Use cached results instead of running the command (if available)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Don't save results to cache")
    parser.add_argument("--force-cache", action="store_true",
                        help="Cache results even if command exits with non-zero code")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Clear cached results (for given command or all if no command specified)")
    parser.add_argument("--list-cache", action="store_true",
                        help="List all cached results")
    
    args = parser.parse_args()
    
    # Handle cache listing
    if args.list_cache:
        if CACHE_DIR.exists():
            cache_files = list(CACHE_DIR.glob("*.json"))
            if cache_files:
                print(f"Cached results in {CACHE_DIR}:")
                for f in sorted(cache_files):
                    try:
                        with open(f) as cf:
                            data = json.load(cf)
                        cmd = data.get('command', 'N/A')[:60]
                        cached_at = data.get('cached_at', 'N/A')
                        print(f"  {f.stem}: {cmd}...")
                        print(f"    Cached: {cached_at}, Peak USS: {format_memory(data.get('peak_uss', 0))}")
                    except (json.JSONDecodeError, IOError):
                        print(f"  {f.stem}: (corrupted)")
            else:
                print("No cached results found.")
        else:
            print("No cached results found.")
        sys.exit(0)
    
    # Handle cache clearing
    if args.clear_cache:
        count = clear_cache(args.command)
        if args.command:
            print(f"Cleared cache for command: {args.command}" if count else "No cache found for command.")
        else:
            print(f"Cleared {count} cached result(s).")
        sys.exit(0)
    
    if args.pid:
        # Monitor existing process
        if not psutil.pid_exists(args.pid):
            print(f"Error: PID {args.pid} does not exist")
            sys.exit(1)
        monitor_process(args.pid, args.interval, args.details, args.group_threshold)
    elif args.command:
        # Check for cached results first if requested
        if args.use_cached:
            cached = load_cache(args.command)
            if cached:
                print(f"Using cached results for: {args.command}")
                print_cached_results(cached, args.details, args.group_threshold)
                sys.exit(cached.get('exit_code', 0))
            else:
                print(f"No cached results found for command. Running command...")
        
        # Run command and monitor
        exit_code = run_and_monitor(
            args.command, 
            args.interval, 
            args.details, 
            args.group_threshold,
            save_to_cache=not args.no_cache,
            force_cache=args.force_cache
        )
        sys.exit(exit_code)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
