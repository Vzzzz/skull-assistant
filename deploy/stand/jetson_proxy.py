#!/usr/bin/env python3
"""Minimal HTTP/HTTPS forward proxy for the Jetson over the USB link (JETSON_BRINGUP §0a), stdlib only.

The board has no internet of its own (no Wi-Fi by the owner's choice, ICS on Windows breaks the VPN path of the
WSL agents). Run this in WSL, then on the board point apt / curl / pip / git / hf at it through a reverse tunnel:

  WSL:    .venv/bin/python deploy/stand/jetson_proxy.py --port 3128 &
          ssh -N -R 3128:127.0.0.1:3128 jetson &                 # keep this ssh running while the board needs the net
  Jetson: export http_proxy=http://127.0.0.1:3128 https_proxy=http://127.0.0.1:3128 no_proxy=localhost,127.0.0.1
          sudo -E apt update                                      # or: echo 'Acquire::http::Proxy "http://127.0.0.1:3128";
                                                                  #      Acquire::https::Proxy "http://127.0.0.1:3128";' | sudo tee /etc/apt/apt.conf.d/90proxy

Handles CONNECT (TLS passthrough) and plain absolute-URI GET/HEAD/POST. Outgoing connections leave from WSL, so they
follow the laptop's routing (VPN included). Single-threaded per connection via threads; no auth, listens on 127.0.0.1 only.
"""
import argparse
import select
import socket
import threading
from urllib.parse import urlsplit

BUF = 65536


def pipe(a, b):
    socks = [a, b]
    try:
        while True:
            r, _, x = select.select(socks, [], socks, 300)
            if x or not r:
                break
            for s in r:
                data = s.recv(BUF)
                if not data:
                    return
                (b if s is a else a).sendall(data)
    except OSError:
        pass


def handle(client):
    try:
        client.settimeout(30)
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = client.recv(BUF)
            if not chunk:
                return
            head += chunk
        header, _, rest = head.partition(b"\r\n\r\n")
        lines = header.decode("latin-1").split("\r\n")
        method, target, version = lines[0].split(" ", 2)
        if method == "CONNECT":
            host, _, port = target.partition(":")
            upstream = socket.create_connection((host, int(port or 443)), timeout=30)
            client.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            if rest:
                upstream.sendall(rest)
        else:
            u = urlsplit(target)
            host, port = u.hostname, u.port or 80
            path = (u.path or "/") + (("?" + u.query) if u.query else "")
            upstream = socket.create_connection((host, port), timeout=30)
            out = [f"{method} {path} {version}"] + [l for l in lines[1:] if not l.lower().startswith("proxy-")]
            upstream.sendall(("\r\n".join(out) + "\r\n\r\n").encode("latin-1") + rest)
        client.settimeout(None)
        upstream.settimeout(None)
        pipe(client, upstream)
        upstream.close()
    except Exception as e:  # one bad request must not take the proxy down
        try:
            client.sendall(f"HTTP/1.1 502 Bad Gateway\r\n\r\n{e}\r\n".encode())
        except OSError:
            pass
    finally:
        client.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=3128)
    ap.add_argument("--bind", default="127.0.0.1")
    a = ap.parse_args()
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((a.bind, a.port))
    srv.listen(64)
    print(f"jetson_proxy: listening on {a.bind}:{a.port}", flush=True)
    while True:
        c, _ = srv.accept()
        threading.Thread(target=handle, args=(c,), daemon=True).start()


if __name__ == "__main__":
    main()
