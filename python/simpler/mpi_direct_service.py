# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Rank 0 controller and P/D service ranks for the MPI Direct supervisor."""

from __future__ import annotations

import argparse
import contextlib
import hmac
import json
import os
import queue
import secrets
import select
import signal
import socket
import struct
import subprocess
import threading
import time
import traceback
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path

from .mpi_direct_protocol import MPI_DIRECT_STARTUP_TOKEN_ENV
from .mpi_direct_runtime import _import_mpi, _launcher_rank, _pre_mpi_gate, _receive_rank_startup
from .mpi_direct_service_relay import LOOPBACK, Relay, close_socket, listen
from .mpi_direct_topology import _is_loopback_host, _positive_finite

ROUTER_TOKEN_ENV = "SIMPLER_MPI_DIRECT_ROUTER_TOKEN"
_ENV_PREFIX = "SIMPLER_MPI_DIRECT_"
_ROLES = ("prefill", "decode")
_SERVICE_TAG = 10
_FRAME_SIZE = struct.Struct("!I")
_MAX_FRAME_BYTES = 8 * 1024 * 1024
_CHILD_EXIT_GRACE_S = 3.0
_LAUNCHER_ENV_PREFIXES = ("PMI_", "PMIX_", "OMPI_", "I_MPI_", "HYDRA_", "MPI_LOCAL", _ENV_PREFIX)


def _port(value, field):
    if type(value) is not int or not 0 < value <= 65535:
        raise ValueError(f"{field} must be a TCP port in 1..65535")
    return value


@dataclass(frozen=True)
class ServiceConfig:
    controller_host: str
    router_port: int
    services: tuple[dict, ...]
    startup_timeout_s: float
    session_timeout_s: float
    launcher_args: tuple[str, ...]

    @property
    def hosts(self) -> tuple[str, ...]:
        return (self.controller_host, *(service["host"] for service in self.services))

    @property
    def world_size(self) -> int:
        return len(self.services) + 1

    @classmethod
    def load(cls, path: str) -> ServiceConfig:  # noqa: PLR0912 -- validate process placement before launch
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("services"), list):
            raise ValueError("service configuration requires a services array")
        services = tuple(data["services"])
        seen_ids = set()
        seen_ports = set()
        roles = set()
        for spec in services:
            if not isinstance(spec, dict):
                raise ValueError("each service must be an object")
            instance_id, role, host, argv = (spec.get(field) for field in ("id", "role", "host", "argv"))
            if not isinstance(instance_id, str) or not instance_id or instance_id in seen_ids:
                raise ValueError("service IDs must be nonempty and unique")
            if role not in _ROLES or not isinstance(host, str) or not host:
                raise ValueError(f"invalid role or host for service {instance_id!r}")
            if not isinstance(argv, list) or not argv or not all(isinstance(arg, str) for arg in argv) or not argv[0]:
                raise ValueError(f"service {instance_id!r} requires a nonempty argv list")
            if spec.get("cwd") is not None and not isinstance(spec["cwd"], str):
                raise ValueError(f"service {instance_id!r} cwd must be a string")
            env = spec.get("env", {})
            if not isinstance(env, dict) or not all(isinstance(value, str) for value in env.values()):
                raise ValueError(f"service {instance_id!r} env must be a string dictionary")
            if role == "decode":
                endpoint = (host, _port(spec.get("bind_port"), f"{instance_id}.bind_port"))
                if endpoint in seen_ports:
                    raise ValueError("decode endpoints on a host must use distinct bind_port values")
                seen_ports.add(endpoint)
            elif "bind_port" in spec:
                raise ValueError("bind_port is only valid for decode services")
            seen_ids.add(instance_id)
            roles.add(role)
        if roles != set(_ROLES):
            raise ValueError("services must contain at least one prefill and decode instance")
        host = data.get("controller_host", LOOPBACK)
        router_port = _port(data.get("router_port"), "router_port")
        if not isinstance(host, str) or not host:
            raise ValueError("controller_host must be a nonempty string")
        if (host, router_port) in seen_ports:
            raise ValueError("router_port and decode bind_port must differ on the controller host")
        args = data.get("launcher_args", [])
        if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
            raise ValueError("launcher_args must be a string list")
        cfg = cls(
            host,
            router_port,
            services,
            _positive_finite(data.get("startup_timeout_s", 120), "startup_timeout_s"),
            _positive_finite(data.get("session_timeout_s", 30), "session_timeout_s"),
            tuple(args),
        )
        if len(set(cfg.hosts)) > 1 and _is_loopback_host(host):
            raise ValueError("controller_host must be reachable by all MPI ranks in a multi-host job")
        host_groups = [host for host, _ in groupby(cfg.hosts)]
        if len(set(host_groups)) != len(host_groups):
            raise ValueError("hosts must be contiguous in rank order")
        return cfg

    def records(self, router_token: str) -> dict[int, dict]:
        identities = [{"id": spec["id"], "role": spec["role"]} for spec in self.services]
        common = {
            "identities": identities,
            "session_timeout_s": self.session_timeout_s,
            "startup_timeout_s": self.startup_timeout_s,
        }
        return {
            0: {**common, "port": self.router_port, "token": router_token},
            **{rank: {**common, "service": spec} for rank, spec in enumerate(self.services, 1)},
        }


def _read_exact(sock: socket.socket, length: int) -> bytearray:
    data = bytearray()
    while len(data) < length:
        chunk = sock.recv(length - len(data))
        if not chunk:
            raise ConnectionError("local control peer closed")
        data.extend(chunk)
    return data


def _read_frame(sock: socket.socket) -> dict:
    (size,) = _FRAME_SIZE.unpack(_read_exact(sock, _FRAME_SIZE.size))
    if not 0 < size <= _MAX_FRAME_BYTES:
        raise ValueError("invalid local control frame size")
    frame = json.loads(_read_exact(sock, size).decode("utf-8"))
    if not isinstance(frame, dict):
        raise ValueError("local control frame must be a JSON object")
    return frame


def _write_frame(sock: socket.socket, frame: dict) -> None:
    data = json.dumps(frame, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(data) > _MAX_FRAME_BYTES:
        raise ValueError("local control frame exceeds size limit")
    sock.sendall(_FRAME_SIZE.pack(len(data)) + data)


class _LocalControl:
    def __init__(self, listener, rank, token, events, timeout) -> None:
        self.listener = listener
        self.rank = rank
        self.token = token
        self.events = events
        self.timeout = timeout
        self.peer = None
        self.closed = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._read, daemon=True).start()

    def _authenticate(self):
        while not self.closed.is_set():
            try:
                peer, _ = self.listener.accept()
            except OSError:
                return None
            peer.settimeout(self.timeout)
            try:
                msg = _read_frame(peer)
                token = msg.get("token")
                if msg.get("type") == "hello" and isinstance(token, str):
                    if hmac.compare_digest(token.encode(), self.token.encode()):
                        _write_frame(peer, {"type": "hello", "rank": self.rank})
                        return peer
            except (OSError, ValueError):
                pass
            peer.close()
        return None

    def _read(self) -> None:
        peer = self._authenticate()
        if peer is None:
            return
        self.events.put(("connected", peer))
        try:
            while not self.closed.is_set():
                select.select([peer], [], [])
                self.events.put(("ipc", _read_frame(peer)))
        except (OSError, ValueError):
            pass
        finally:
            self.events.put(("disconnected",))
            close_socket(peer)

    def send(self, message: dict) -> bool:
        if self.peer is None:
            return False
        try:
            _write_frame(self.peer, message)
            return True
        except (OSError, ValueError):
            close_socket(self.peer)
            self.peer = None
            return False

    def close(self) -> None:
        self.closed.set()
        close_socket(self.listener)
        if self.peer is not None:
            close_socket(self.peer)


class _Rank:
    def __init__(self, rank, record, port, token) -> None:
        self.rank = rank
        self.identities = dict(enumerate(record["identities"], 1))
        self.members = {
            "all": tuple(self.identities),
            **{role: tuple(rank for rank, spec in self.identities.items() if spec["role"] == role) for role in _ROLES},
        }
        self.events = queue.Queue()
        self.timeout = record["session_timeout_s"]
        self.startup_timeout = record["startup_timeout_s"]
        self.control = _LocalControl(listen(port), rank, token, self.events, min(self.timeout, 5))
        self.relay = None

    def start(self, mpi) -> None:
        self.mpi = mpi
        self.world = mpi.COMM_WORLD
        self.comms = {"all": self.world}
        for color, role in enumerate(_ROLES):
            members = (0, *self.members[role])
            self.comms[role] = self.world.Split(color if self.rank in members else mpi.UNDEFINED, self.rank)
        self.receiver = threading.Thread(target=self._receive, daemon=True)
        self.receiver.start()
        self.control.start()

    def _receive(self) -> None:
        try:
            status = self.mpi.Status()
            while True:
                kind, key, payload = self.world.recv(source=self.mpi.ANY_SOURCE, tag=_SERVICE_TAG, status=status)
                source = status.Get_source()
                if source == self.rank and kind == "unblock":
                    return
                if kind == "stream_data":
                    self.relay.deliver(source, key, payload)
                else:
                    self.events.put(("mpi", source, kind, key, payload))
        except Exception as exc:
            self.events.put(("error", exc))

    def send(self, rank, kind, key=None, payload=None) -> None:
        self.world.send((kind, key, payload), dest=rank, tag=_SERVICE_TAG)

    def finish_mpi(self) -> None:
        self.world.Barrier()
        self.send(self.rank, "unblock")
        self.receiver.join(timeout=self.timeout)
        if self.receiver.is_alive():
            raise TimeoutError("MPI service receiver did not stop")
        for role in _ROLES:
            if self.comms[role] != self.mpi.COMM_NULL:
                self.comms[role].Free()

    def next_event(self, deadline=None):
        try:
            if deadline is None:
                return self.events.get()
            return self.events.get(timeout=max(0, deadline - time.monotonic()))
        except queue.Empty:
            return ("timeout",)

    def close(self) -> None:
        self.control.close()


class _Controller(_Rank):
    def __init__(self, rank, record) -> None:
        super().__init__(rank, record, record["port"], record["token"])
        self.latest = {}
        self.exited = set()
        self.ending = False
        self.ready = False
        self.exit_code = 0
        self.shutdown_id = None

    def _request(self, frame, deadline):
        kind, msg_id = frame.get("type"), frame.get("id")
        if not isinstance(msg_id, str) or not msg_id:
            return deadline
        if self.ending:
            self.control.send({"type": "response", "id": msg_id, "ok": False, "error": "job is stopping"})
        elif kind == "shutdown":
            self.shutdown_id = msg_id
            return self._begin_shutdown()
        elif kind == "broadcast":
            self._broadcast(frame)
        else:
            self.control.send({"type": "response", "id": msg_id, "ok": False, "error": "unknown command"})
        return deadline

    def _broadcast(self, frame) -> None:
        scope = frame.get("scope")
        ranks = self.members.get(scope) if isinstance(scope, str) else None
        response = {"type": "response", "id": frame["id"], "ok": False}
        if ranks is None:
            response["error"] = "invalid broadcast scope"
        elif any(self.latest.get(rank, {}).get("state") != "ready" for rank in ranks):
            response["error"] = "broadcast scope contains a service that is not ready"
        else:
            for rank in ranks:
                self.send(rank, "broadcast", scope)
            comm = self.comms[scope]
            comm.bcast(frame, root=0)
            comm.Barrier()
            response.update(ok=True, payload={"ranks": list(ranks)})
        self.control.send(response)

    def _begin_shutdown(self):
        self.ending = True
        return self._stop_group("prefill")

    def _stop_group(self, role):
        ranks = set(self.members[role]) - self.exited
        for rank in sorted(ranks):
            self.send(rank, "shutdown")
        return time.monotonic() + self.timeout + 3 * _CHILD_EXIT_GRACE_S

    def _status(self, rank, state, payload) -> None:
        spec = self.identities[rank]
        frame = {
            "type": "status",
            "rank": rank,
            "instance_id": spec["id"],
            "role": spec["role"],
            "state": state,
            "payload": payload,
        }
        self.latest[rank] = frame
        self.control.send(frame)
        if state == "exited":
            self.exited.add(rank)
            if not self.ending or payload["returncode"] != 0:
                self.exit_code = 1
        elif state == "failed":
            self.exit_code = 1

    def _update_startup(self, deadline):
        if self.ending or self.control.peer is None:
            return deadline
        if self.exit_code:
            return self._begin_shutdown()
        if not self.ready and all(self.latest.get(rank, {}).get("state") == "ready" for rank in self.identities):
            self.ready = True
            return None
        return deadline

    def run(self) -> int:
        deadline = time.monotonic() + self.startup_timeout
        decode_stopping = False
        while len(self.exited) < len(self.identities) or not self.ending:
            kind, *parts = self.next_event(deadline)
            if kind == "connected":
                self.control.peer = parts[0]
                for frame in self.latest.values():
                    self.control.send(frame)
            elif kind == "ipc":
                deadline = self._request(parts[0], deadline)
            elif kind in ("timeout", "disconnected"):
                if kind == "disconnected":
                    self.control.peer = None
                if not self.ending:
                    self.exit_code = 1
                    deadline = self._begin_shutdown()
                elif kind == "timeout":
                    raise TimeoutError("timed out waiting for service processes to exit")
            elif kind == "mpi":
                rank, name, key, payload = parts
                if name == "status":
                    self._status(rank, key, payload)
            elif kind == "error":
                raise parts[0]

            deadline = self._update_startup(deadline)
            if self.ending and not decode_stopping and self.exited.issuperset(self.members["prefill"]):
                deadline = self._stop_group("decode")
                decode_stopping = True

        if self.shutdown_id is not None:
            reply = {"type": "response", "id": self.shutdown_id, "ok": self.exit_code == 0}
            if self.exit_code:
                reply["error"] = "a service exited with an error"
            self.control.send(reply)
        for rank in self.identities:
            self.send(rank, "finish")
        return self.exit_code


def _terminate_child(child) -> None:
    if child.poll() is not None:
        return
    if os.name == "nt":
        child.terminate()
    else:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=_CHILD_EXIT_GRACE_S)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            child.kill()
        else:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=_CHILD_EXIT_GRACE_S)


class _Service(_Rank):
    def __init__(self, rank, record) -> None:
        spec = record["service"]
        token = secrets.token_urlsafe(32)
        super().__init__(rank, record, 0, token)
        role = spec["role"]
        peers = {rank: self.identities[rank]["id"] for rank in self.members["decode"]} if role == "prefill" else {}
        self.child = None
        try:
            self.relay = Relay(role, spec.get("bind_port"), peers, self.events, min(self.timeout, 5))
            self.child = self._start_child(spec, token)
        except BaseException:
            self.close()
            raise

    def _start_child(self, spec, token):
        env = {name: value for name, value in os.environ.items() if not name.upper().startswith(_LAUNCHER_ENV_PREFIXES)}
        env.update(spec.get("env", {}))
        options = {
            "HOST": LOOPBACK,
            "PORT": str(self.control.listener.getsockname()[1]),
            "TOKEN": token,
            "RANK": str(self.rank),
            "INSTANCE_ID": spec["id"],
            "ROLE": spec["role"],
        }
        if spec["role"] == "prefill":
            options["PEERS"] = json.dumps(self.relay.ports(), separators=(",", ":"))
        else:
            options["BIND_PORT"] = str(spec["bind_port"])
        env.update({_ENV_PREFIX + name: value for name, value in options.items()})
        return subprocess.Popen(
            spec["argv"], env=env, cwd=spec.get("cwd"), stdin=subprocess.DEVNULL, start_new_session=os.name != "nt"
        )

    def _watch_child(self, child) -> None:
        self.events.put(("child_exit", child.wait()))

    def _status(self, state, payload) -> None:
        self.send(0, "status", state, payload)

    def _on_message(self, source, kind, key, payload):
        if source == 0 and kind == "broadcast":
            comm = self.comms[key]
            self.control.send(comm.bcast(None, root=0))
            comm.Barrier()
        elif kind == "stream_open" and source in self.members["prefill"]:
            self.relay.on_open(source, key)
        elif kind == "stream_opened" and source in self.members["decode"]:
            self.relay.on_opened(source, key)
        elif kind == "stream_reset":
            self.relay.on_reset(source, key)

    def run(self) -> int:  # noqa: PLR0912 -- process, MPI and IPC events share the rank's lifecycle
        deadline = None
        stopping = False
        exited = False
        child = self.child
        assert child is not None
        self.relay.start(self.send)
        self._status("starting", {"pid": child.pid})
        threading.Thread(target=self._watch_child, args=(child,), daemon=True).start()
        while True:
            kind, *parts = self.next_event(deadline)
            if kind == "connected":
                self.control.peer = parts[0]
            elif kind == "disconnected":
                self.control.peer = None
                if not stopping and not exited:
                    self._status("disconnected", {})
            elif kind == "ipc" and not exited:
                frame = parts[0]
                state, payload = frame.get("state"), frame.get("payload", {})
                if frame.get("type") == "status" and state in ("ready", "draining", "failed"):
                    if isinstance(payload, dict):
                        self._status(state, payload)
            elif kind == "child_exit":
                exited = True
                deadline = None
                self.relay.close()
                self.control.close()
                self._status("exited", {"returncode": parts[0]})
            elif kind == "timeout":
                deadline = None
                _terminate_child(child)
            elif kind == "relay_accept":
                if exited:
                    parts[1].close()
                else:
                    self.relay.accept(*parts)
            elif kind in ("relay_eof", "relay_reset"):
                self.relay.finish(parts[0], kind == "relay_reset")
            elif kind == "mpi":
                source, command, key, payload = parts
                if source == 0 and command == "finish":
                    return 0
                if source == 0 and command == "shutdown" and not exited:
                    stopping = True
                    if self.control.send({"type": "shutdown"}):
                        deadline = time.monotonic() + self.timeout
                    else:
                        _terminate_child(child)
                elif not exited or command == "broadcast":
                    self._on_message(source, command, key, payload)
                elif command == "stream_open":
                    self.send(source, "stream_reset", key)
            elif kind == "error":
                raise parts[0]

    def close(self) -> None:
        super().close()
        if self.relay is not None:
            self.relay.close()
        if self.child is not None:
            _terminate_child(self.child)


def _run(args) -> int:
    rank = _launcher_rank()
    token = os.environ[MPI_DIRECT_STARTUP_TOKEN_ENV]
    deadline = time.monotonic() + args.startup_timeout
    record, deadline = _receive_rank_startup(args.startup_host, args.startup_port, token, rank, deadline)
    owner = None
    try:
        mpi = _import_mpi()
        owner = _Controller(rank, record) if rank == 0 else _Service(rank, record)
    except Exception as exc:
        _pre_mpi_gate(args.startup_host, args.startup_port, token, rank, args.startup_timeout, exc, deadline=deadline)
        raise

    def cleanup() -> bool:
        owner.close()
        return True

    try:
        _pre_mpi_gate(
            args.startup_host, args.startup_port, token, rank, args.startup_timeout, deadline=deadline, cleanup=cleanup
        )
        provided = mpi.Init_thread(required=mpi.THREAD_MULTIPLE)
        world = mpi.COMM_WORLD
        try:
            if provided < mpi.THREAD_MULTIPLE:
                raise RuntimeError("MPI Direct service ranks require MPI_THREAD_MULTIPLE")
            if world.Get_rank() != rank or world.Get_size() != len(owner.identities) + 1:
                raise RuntimeError("MPI_COMM_WORLD does not match service configuration")
            world.Set_errhandler(mpi.ERRORS_RETURN)
            owner.start(mpi)
            result = owner.run()
            owner.finish_mpi()
        except BaseException:
            traceback.print_exc()
            with contextlib.suppress(Exception):
                owner.close()
            world.Abort(1)
            raise
        mpi.Finalize()
        return result
    finally:
        owner.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--startup-host", required=True)
    parser.add_argument("--startup-port", type=int, required=True)
    parser.add_argument("--startup-timeout", type=float, required=True)
    args = parser.parse_args(argv)

    def terminate(signum, _frame):
        raise SystemExit(128 + signum)

    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        return _run(args)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    raise SystemExit(main())
