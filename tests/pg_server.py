"""Throwaway local PostgreSQL server for testing the durable research store (skipped if unavailable)."""

import glob
import os
import shutil
import socket
import subprocess
import tempfile
import time


def _bin(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    candidates = sorted(glob.glob(f"/usr/lib/postgresql/*/bin/{name}"))
    return candidates[-1] if candidates else None


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class PgServer:
    def __init__(self):
        self.initdb, self.pg_ctl = _bin("initdb"), _bin("pg_ctl")
        if not self.initdb or not self.pg_ctl:
            raise RuntimeError("PostgreSQL server binaries not found")
        self.dir = tempfile.mkdtemp(prefix="pgtest-")
        os.chmod(self.dir, 0o777)
        self.data = os.path.join(self.dir, "data")
        self.port = _free_port()
        self.prefix = ["runuser", "-u", "postgres", "--"] if os.geteuid() == 0 else []

    def _run(self, *args):
        subprocess.run(self.prefix + list(args), check=True, capture_output=True, timeout=120)

    def start(self) -> str:
        self._run(self.initdb, "-D", self.data, "-U", "pgtest", "--auth=trust", "-E", "UTF8", "--no-instructions")
        self._run(self.pg_ctl, "-D", self.data, "-w", "-l", os.path.join(self.dir, "log"), "-o",
                  f"-p {self.port} -k {self.dir} -c listen_addresses=127.0.0.1", "start")
        dsn = f"postgresql://pgtest@127.0.0.1:{self.port}/postgres"
        deadline = time.time() + 30
        import psycopg

        while True:
            try:
                psycopg.connect(dsn, connect_timeout=2).close()
                return dsn
            except psycopg.OperationalError:
                if time.time() > deadline:
                    raise
                time.sleep(0.3)

    def stop(self):
        try:
            self._run(self.pg_ctl, "-D", self.data, "-m", "immediate", "stop")
        except Exception:  # noqa: BLE001
            pass
        shutil.rmtree(self.dir, ignore_errors=True)
