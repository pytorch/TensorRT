import argparse
import logging
import os
import signal
import subprocess
import sys

from torch_tensorrt.distributed._nccl_utils import (
    ensure_nccl_symlink,
    get_nccl_library_path,
)

logger = logging.getLogger("torchtrtrun")


def _setup_nccl_env(env: dict[str, str]) -> dict[str, str]:
    """Return a copy of env with LD_LIBRARY_PATH and LD_PRELOAD wired for NCCL."""
    lib_dir = get_nccl_library_path()
    if lib_dir is None:
        logger.info("System NCCL detected — no setup needed.")
        return env

    ensure_nccl_symlink(lib_dir)

    env = dict(env)

    # LD_LIBRARY_PATH — lets dlopen find libnccl.so by name
    ld_lib = env.get("LD_LIBRARY_PATH", "")
    if lib_dir not in ld_lib:
        env["LD_LIBRARY_PATH"] = f"{lib_dir}:{ld_lib}" if ld_lib else lib_dir

    # LD_PRELOAD — forces libnccl.so.2 into the process before TRT's loader
    # runs, so any subsequent dlopen("libnccl.so") finds it already resident.
    nccl_so2 = os.path.join(lib_dir, "libnccl.so.2")
    if os.path.exists(nccl_so2):
        ld_pre = env.get("LD_PRELOAD", "")
        if nccl_so2 not in ld_pre:
            env["LD_PRELOAD"] = f"{nccl_so2}:{ld_pre}" if ld_pre else nccl_so2

    logger.debug("NCCL lib dir    : %s", lib_dir)
    logger.debug("LD_LIBRARY_PATH : %s", env["LD_LIBRARY_PATH"])
    logger.debug("LD_PRELOAD      : %s", env.get("LD_PRELOAD", ""))

    return env


def _parse_rdzv_endpoint(endpoint: str) -> tuple[str, int]:
    """Parse host:port from a rendezvous endpoint string."""
    if not endpoint:
        return "127.0.0.1", 29500
    if ":" in endpoint:
        host, port_str = endpoint.rsplit(":", 1)
        try:
            return host, int(port_str)
        except ValueError:
            raise ValueError(f"Invalid --rdzv_endpoint port in '{endpoint}'")
    return endpoint, 29500


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="torchtrtrun",
        description="torchrun-compatible launcher with automatic NCCL setup for TRT.",
        usage="torchtrtrun [options] script.py [script_args...]",
    )

    # Worker count / topology (matches torchrun)
    parser.add_argument(
        "--nproc_per_node",
        type=int,
        default=1,
        metavar="NPROC",
        help="Number of worker processes per node (default: 1)",
    )
    parser.add_argument(
        "--nnodes",
        type=int,
        default=1,
        help="Total number of nodes (default: 1)",
    )
    parser.add_argument(
        "--node_rank",
        type=int,
        default=0,
        help="Rank of this node (default: 0)",
    )

    # Rendezvous — torchrun-style (preferred) or legacy --master_addr/port
    rdzv = parser.add_argument_group("rendezvous")
    rdzv.add_argument(
        "--rdzv_endpoint",
        default="",
        metavar="HOST:PORT",
        help="Rendezvous endpoint in host:port format, e.g. 169.254.204.57:29500",
    )
    rdzv.add_argument(
        "--rdzv_backend",
        default="static",
        help="Rendezvous backend (default: static)",
    )
    rdzv.add_argument(
        "--rdzv_id",
        default="none",
        help="Rendezvous job id (default: none)",
    )

    # Legacy aliases kept for compatibility
    legacy = parser.add_argument_group("legacy (use --rdzv_endpoint instead)")
    legacy.add_argument("--master_addr", default="", help="Master node address")
    legacy.add_argument("--master_port", type=int, default=0, help="Master node port")

    parser.add_argument("training_script", help="Path to the script to run")
    parser.add_argument("training_script_args", nargs=argparse.REMAINDER)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[torchtrtrun] %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    # Resolve master addr/port: rdzv_endpoint takes priority over legacy flags
    if args.rdzv_endpoint:
        master_addr, master_port = _parse_rdzv_endpoint(args.rdzv_endpoint)
    elif args.master_addr:
        master_addr = args.master_addr
        master_port = args.master_port or 29500
    else:
        master_addr = "127.0.0.1"
        master_port = args.master_port or 29500

    world_size = args.nnodes * args.nproc_per_node
    env = _setup_nccl_env(os.environ.copy())
    env["MASTER_ADDR"] = master_addr
    env["MASTER_PORT"] = str(master_port)
    env["WORLD_SIZE"] = str(world_size)

    procs = []
    for local_rank in range(args.nproc_per_node):
        global_rank = args.node_rank * args.nproc_per_node + local_rank
        worker_env = dict(env)
        worker_env["RANK"] = str(global_rank)
        worker_env["LOCAL_RANK"] = str(local_rank)

        cmd = [sys.executable, args.training_script] + args.training_script_args
        p = subprocess.Popen(cmd, env=worker_env)
        procs.append(p)
        logger.info(
            "Spawned rank %d (local_rank=%d, pid=%d)", global_rank, local_rank, p.pid
        )

    def _signal_all(sig: int, frame: object) -> None:
        for p in procs:
            try:
                p.send_signal(sig)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGINT, _signal_all)
    signal.signal(signal.SIGTERM, _signal_all)

    exit_codes = [p.wait() for p in procs]
    failed = [(i, c) for i, c in enumerate(exit_codes) if c != 0]
    if failed:
        for rank_idx, code in failed:
            if code < 0:
                try:
                    sig = signal.Signals(-code)
                    logger.error("Worker rank %d killed by %s", rank_idx, sig.name)
                except ValueError:
                    logger.error("Worker rank %d killed by signal %d", rank_idx, -code)
            else:
                logger.error("Worker rank %d exited with code %d", rank_idx, code)
        sys.exit(max(c for _, c in failed))
