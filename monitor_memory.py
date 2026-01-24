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
import subprocess
import sys
import time
from datetime import datetime
from typing import Optional, Tuple, List

try:
    import psutil
except ImportError:
    print("Error: psutil is required. Install with: pip install psutil")
    sys.exit(1)


def get_memory_info(pid: int) -> Tuple[float, float, float, int, List[dict]]:
    """
    Get memory info for a process and all its children.
    
    Returns:
        (total_rss_mb, total_uss_mb, total_pss_mb, num_processes, process_details)
    """
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return 0, 0, 0, 0, []
    
    processes = [parent] + list(parent.children(recursive=True))
    
    total_rss = 0
    total_uss = 0
    total_pss = 0
    details = []
    
    for proc in processes:
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
            except (psutil.AccessDenied, AttributeError):
                uss = 0
                pss = 0
            
            details.append({
                'pid': proc.pid,
                'name': proc.name()[:20],
                'rss_mb': rss / (1024 * 1024),
                'uss_mb': uss / (1024 * 1024),
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    return (
        total_rss / (1024 * 1024),
        total_uss / (1024 * 1024),
        total_pss / (1024 * 1024),
        len(details),
        details
    )


def format_memory(mb: float) -> str:
    """Format memory size with appropriate unit."""
    if mb >= 1024:
        return f"{mb/1024:.2f} GB"
    return f"{mb:.1f} MB"


def clear_line():
    """Clear current line in terminal."""
    print("\033[2K\033[G", end="")


def move_up(n: int):
    """Move cursor up n lines."""
    if n > 0:
        print(f"\033[{n}A", end="")


def monitor_process(pid: int, interval: float = 0.5, show_details: bool = False):
    """Monitor memory usage of a process and its children."""
    peak_rss = 0
    peak_uss = 0
    peak_processes = 0
    start_time = time.time()
    last_lines = 0
    
    print(f"\n{'='*60}")
    print(f"Monitoring PID {pid} and children (Ctrl+C to stop)")
    print(f"{'='*60}\n")
    
    try:
        while True:
            rss, uss, pss, num_procs, details = get_memory_info(pid)
            
            if num_procs == 0:
                print("\nProcess ended.")
                break
            
            # Update peaks
            peak_rss = max(peak_rss, rss)
            peak_uss = max(peak_uss, uss)
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
                lines.append(f"  Current PSS:  {format_memory(pss):>12}")
            lines.append(f"")
            
            if show_details and details:
                # Sort by RSS descending, show top 10
                sorted_details = sorted(details, key=lambda x: x['rss_mb'], reverse=True)[:10]
                lines.append(f"  Top processes by memory:")
                lines.append(f"  {'PID':>8}  {'Name':<20}  {'RSS':>10}  {'USS':>10}")
                lines.append(f"  {'-'*8}  {'-'*20}  {'-'*10}  {'-'*10}")
                for d in sorted_details:
                    lines.append(f"  {d['pid']:>8}  {d['name']:<20}  {format_memory(d['rss_mb']):>10}  {format_memory(d['uss_mb']):>10}")
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
    print(f"  Peak processes:  {peak_processes}")
    print(f"{'='*60}\n")


def run_and_monitor(command: str, interval: float = 0.5, show_details: bool = False):
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
    peak_processes = 0
    start_time = time.time()
    samples = []
    
    print(f"\n{'='*60}")
    print(f"Monitoring PID {proc.pid} (command running in foreground)")
    print(f"Memory stats will be shown after command completes")
    print(f"{'='*60}\n")
    
    try:
        while proc.poll() is None:
            rss, uss, pss, num_procs, details = get_memory_info(proc.pid)
            
            if num_procs > 0:
                peak_rss = max(peak_rss, rss)
                peak_uss = max(peak_uss, uss)
                peak_processes = max(peak_processes, num_procs)
                samples.append((time.time() - start_time, rss, uss, num_procs))
            
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
        print(f"  ")
        print(f"  Note: USS (Unique Set Size) is memory unique to each process.")
        print(f"        Low USS vs RSS indicates good copy-on-write sharing.")
    print(f"  Peak processes:  {peak_processes}")
    print(f"  Samples taken:   {len(samples)}")
    print(f"{'='*60}\n")
    
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
        """
    )
    parser.add_argument("command", nargs="?", help="Command to run and monitor")
    parser.add_argument("--pid", type=int, help="PID of existing process to monitor")
    parser.add_argument("--interval", "-i", type=float, default=0.5, 
                        help="Sampling interval in seconds (default: 0.5)")
    parser.add_argument("--details", "-d", action="store_true",
                        help="Show per-process memory breakdown")
    
    args = parser.parse_args()
    
    if args.pid:
        # Monitor existing process
        if not psutil.pid_exists(args.pid):
            print(f"Error: PID {args.pid} does not exist")
            sys.exit(1)
        monitor_process(args.pid, args.interval, args.details)
    elif args.command:
        # Run command and monitor
        exit_code = run_and_monitor(args.command, args.interval, args.details)
        sys.exit(exit_code)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
