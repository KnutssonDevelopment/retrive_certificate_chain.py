"""
certificate_chain.py

Production-ready utilities to fetch a server's leaf certificate and reconstruct
its certificate chain *locally* using only the CA certificates available in
the current Python SSLContext (i.e., `ssl.create_default_context()`), without
scanning capaths and without doing any network AIA fetching.

Requires:
    - Python 3.12+
    - cryptography

Typical use:
    from certificate_chain import (
        build_chain_from_url_using_local_store,
        summarize_chain,
        save_chain_pem,
    )

    chain, completed = build_chain_from_url_using_local_store("https://example.com")
    print("Completed locally:", completed)
    for row in summarize_chain(chain):
        print(row)
    save_chain_pem(chain, prefix="example-com")

Design notes:
    - SNI is always used (server_hostname=hostname).
    - We first try a verified handshake; if it fails (e.g., incomplete server chain),
      we retry unverified to still retrieve the peer's leaf for local reconstruction.
    - Local reconstruction only uses CA certificates from `SSLContext.get_ca_certs(binary_form=True)`.
      No capath scanning and no external AIA downloads are performed.
    - Chain building is based on:
        1) Issuer DN match,
        2) AKI(child) == SKI(candidate) when present,
        3) cryptographic signature verification of child by candidate.
    - We stop at a self-signed certificate or when no further issuer can be found.

Security caveat:
    Reconstructing a chain locally does not by itself assert trust. To *validate* trust,
    you still need to verify the chain against a trust store (e.g., via a verified
    SSL handshake or a dedicated validator). This module focuses on reproducibly
    assembling the *structural* chain using your local CA set.
"""

from __future__ import annotations

import logging
import socket
import ssl

from datetime import timezone
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple
from urllib.parse import urlparse

from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, ed448, ec, rsa
from cryptography.hazmat.primitives.asymmetric import padding as asy_padding
from cryptography.x509.oid import ExtensionOID, NameOID


# --------------------------------------------------------------------------------------
# Configuration / logging
# --------------------------------------------------------------------------------------

logger = logging.getLogger(__name__)
# Set this in your app:
# logging.basicConfig(level=logging.INFO)


# --------------------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ChainSummaryRow:
    index: int
    subject_cn: str | None
    issuer_cn: str | None
    subject_dn: str
    issuer_dn: str
    not_before: object
    not_after: object
    self_signed: bool
    aki_hex: str | None
    ski_hex: str | None


class ChainBuildError(RuntimeError):
    """Raised when a server certificate cannot be retrieved at all."""


# --------------------------------------------------------------------------------------
# Internal helpers (loading, naming, matching, verification)
# --------------------------------------------------------------------------------------

def _load_der(der: bytes) -> x509.Certificate:
    return x509.load_der_x509_certificate(der, default_backend())


def _load_pem_or_der(data: bytes) -> x509.Certificate:
    try:
        return x509.load_pem_x509_certificate(data, default_backend())
    except Exception:
        return x509.load_der_x509_certificate(data, default_backend())


def _name_to_cn(name: x509.Name) -> str | None:
    for r in name.rdns:
        for a in r:
            if a.oid == NameOID.COMMON_NAME:
                return a.value
    return None


def _name_to_str(name: x509.Name) -> str:
    return name.rfc4514_string()


def _get_ski_hex(cert: x509.Certificate) -> str | None:
    try:
        ext = cert.extensions.get_extension_for_oid(ExtensionOID.SUBJECT_KEY_IDENTIFIER).value
        if ext.digest:
            return ext.digest.hex()
    except Exception:
        return None
    return None


def _get_aki_hex(cert: x509.Certificate) -> str | None:
    try:
        ext = cert.extensions.get_extension_for_oid(ExtensionOID.AUTHORITY_KEY_IDENTIFIER).value
        if ext.key_identifier:
            return ext.key_identifier.hex()
    except Exception:
        return None
    return None


def _is_self_signed(cert: x509.Certificate) -> bool:
    return cert.subject == cert.issuer


def _is_signed_by(child: x509.Certificate, issuer: x509.Certificate) -> bool:
    """
    Verify that `issuer` signed `child` using the appropriate algorithm.
    Returns True on successful verification, otherwise False.
    """
    pub = issuer.public_key()
    try:
        if isinstance(pub, rsa.RSAPublicKey):
            pub.verify(
                child.signature,
                child.tbs_certificate_bytes,
                asy_padding.PKCS1v15(),
                child.signature_hash_algorithm,
            )
        elif isinstance(pub, ec.EllipticCurvePublicKey):
            pub.verify(
                child.signature,
                child.tbs_certificate_bytes,
                ec.ECDSA(child.signature_hash_algorithm),
            )
        elif isinstance(pub, ed25519.Ed25519PublicKey):
            pub.verify(child.signature, child.tbs_certificate_bytes)
        elif isinstance(pub, ed448.Ed448PublicKey):
            pub.verify(child.signature, child.tbs_certificate_bytes)
        else:
            return False
        return True
    except Exception:
        return False

def _validity_utc(cert):
    # cryptography >= 41 har *_utc properties
    nbf = getattr(cert, "not_valid_before_utc", None)
    naf = getattr(cert, "not_valid_after_utc", None)
    if nbf is None:  # ældre versions fallback
        nbf = cert.not_valid_before.replace(tzinfo=timezone.utc)
    if naf is None:
        naf = cert.not_valid_after.replace(tzinfo=timezone.utc)
    return nbf, naf

# --------------------------------------------------------------------------------------
# 1) Fetch the server-sent chain (leaf + whatever the server actually sends)
# --------------------------------------------------------------------------------------

def fetch_server_sent_chain(url_or_host: str, timeout: float = 5.0) -> List[x509.Certificate]:
    """
    Fetch the server's certificate(s) using SNI.

    We try a verified handshake first (STRICT). If verification fails, we retry
    unverified so we can still retrieve the leaf and any server-sent intermediates.

    Args:
        url_or_host: "https://example.com" or "example.com"
        timeout: socket timeout in seconds

    Returns:
        A list of x509.Certificate objects, leaf first. Might be length 1 if the
        server only sends the leaf.

    Raises:
        ChainBuildError: if we cannot obtain even the leaf certificate.
    """
    parsed = urlparse(url_or_host if "://" in url_or_host else f"https://{url_or_host}")
    host = parsed.hostname or parsed.path
    port = parsed.port or 443
    if not host:
        raise ValueError(f"Invalid URL/host: {url_or_host}")

    def _attempt(verify: bool) -> List[x509.Certificate]:
        if verify:
            ctx = ssl.create_default_context()
            ctx.check_hostname = True
            ctx.verify_mode = ssl.CERT_REQUIRED
        else:
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE

        # ALPN is optional, but harmless and can be useful for diagnostics
        try:
            ctx.set_alpn_protocols(["h2", "http/1.1"])
        except Exception:
            pass

        with socket.create_connection((host, port), timeout=timeout) as raw:
            with ctx.wrap_socket(raw, server_hostname=host) as ss:
                # Prefer verified chain when available (Py 3.12)
                if verify and hasattr(ss, "get_verified_chain"):
                    certs = ss.get_verified_chain()
                    return [_load_der(c.to_der()) for c in certs]

                # Otherwise, fall back to unverified variants
                if hasattr(ss, "get_unverified_chain"):
                    certs = ss.get_unverified_chain()
                    return [_load_der(c.to_der()) for c in certs]

                if hasattr(ss, "get_peer_cert_chain"):
                    certs = ss.get_peer_cert_chain()
                    return [_load_der(c.to_der()) for c in certs]

                # Absolute last resort: leaf only
                leaf_der = ss.getpeercert(True)
                return [x509.load_der_x509_certificate(leaf_der, default_backend())]

    # Try verified, then unverified
    try:
        return _attempt(verify=True)
    except ssl.SSLCertVerificationError as e:
        logger.info("Verified handshake failed, retrying unverified: %s", e)
        try:
            return _attempt(verify=False)
        except Exception as e2:
            raise ChainBuildError(f"Failed to retrieve server certificate(s): {e2}") from e2
    except Exception as e:
        # Some endpoints fail even earlier; retry unverified once
        logger.debug("Verified attempt errored (%r), retrying unverified.", e)
        try:
            return _attempt(verify=False)
        except Exception as e2:
            raise ChainBuildError(f"Failed to retrieve server certificate(s): {e2}") from e2


# --------------------------------------------------------------------------------------
# 2) Load "local" CA certificates *only* from your SSLContext (your requirement)
# --------------------------------------------------------------------------------------

def load_local_store_from_context() -> List[x509.Certificate]:
    """
    Returns the CA certificates from the default SSLContext (trust store) in DER form,
    parsed into x509.Certificate objects.

    Note:
        This is exactly equivalent to your `getcertmeta(...).context.get_ca_certs(binary_form=True)`,
        but without opening a socket or doing reverse DNS. It is independent of any server.

    Returns:
        List of x509.Certificate objects (likely roots; intermediates may be present on some systems).
    """
    ctx = ssl.create_default_context()
    ders = ctx.get_ca_certs(binary_form=True)  # returns a list of DER bytes
    out: List[x509.Certificate] = []
    for der in ders:
        try:
            out.append(_load_der(der))
        except Exception:
            continue
    return out


# --------------------------------------------------------------------------------------
# 3) Reconstruct the chain locally by matching against your local store only
# --------------------------------------------------------------------------------------

def complete_chain_with_local_store(
    server_chain: Sequence[x509.Certificate],
    local_store: Sequence[x509.Certificate],
    *,
    max_depth: int = 10,
) -> Tuple[List[x509.Certificate], bool]:
    """
    Extend the server-sent chain by climbing upwards using only the provided local store:
    - issuer DN must match subject DN
    - prefer AKI(child) == SKI(candidate) when available
    - verify signature cryptographically
    Stops at a self-signed certificate or when no suitable issuer can be found.

    Args:
        server_chain: Leaf-first certificates retrieved from the server.
        local_store: Certificates from your local SSLContext (roots and possibly intermediates).
        max_depth: Safety guard to avoid infinite loops in pathological cases.

    Returns:
        (new_chain, changed)
            new_chain: extended list, leaf first
            changed: True if at least one certificate was added
    """
    if not server_chain:
        return [], False

    # Index local store by subject DN for quick lookup
    by_subject: dict[str, List[x509.Certificate]] = {}
    for c in local_store:
        by_subject.setdefault(_name_to_str(c.subject), []).append(c)

    chain: List[x509.Certificate] = list(server_chain)
    changed = False

    # Track seen fingerprints to avoid duplicates
    def _fp(c: x509.Certificate) -> bytes:
        return c.fingerprint(c.signature_hash_algorithm)

    seen = { _fp(c) for c in chain }

    steps = 0
    while steps < max_depth:
        steps += 1
        tail = chain[-1]
        if _is_self_signed(tail):
            break

        issuer_key = _name_to_str(tail.issuer)
        candidates = by_subject.get(issuer_key, [])
        if not candidates:
            break

        # Prefer exact AKI(child) == SKI(candidate) + signature verification
        tail_aki = _get_aki_hex(tail)
        chosen: x509.Certificate | None = None

        if tail_aki:
            for cand in candidates:
                if _get_ski_hex(cand) == tail_aki and _is_signed_by(tail, cand):
                    chosen = cand
                    break

        # Fallback: any candidate that actually verifies the child's signature
        if not chosen:
            for cand in candidates:
                if _is_signed_by(tail, cand):
                    chosen = cand
                    break

        if not chosen:
            break

        fp = _fp(chosen)
        if fp in seen:
            break  # avoid loops

        chain.append(chosen)
        seen.add(fp)
        changed = True

    return chain, changed


# --------------------------------------------------------------------------------------
# 4) Public convenience: full flow + summaries + saving
# --------------------------------------------------------------------------------------

def build_chain_from_url_using_local_store(
    url_or_host: str,
    *,
    timeout: float = 5.0,
) -> Tuple[List[x509.Certificate], bool]:
    """
    Fetch the leaf/server-sent chain from the remote endpoint and attempt to
    complete it locally using only the certificates available from the default
    SSLContext's CA set (no capaths, no AIA network fetch).

    Args:
        url_or_host: "https://example.com" or "example.com"
        timeout: socket timeout in seconds

    Returns:
        (chain, completed)
            chain: leaf-first assembled chain
            completed: True if local completion added at least one certificate
    """
    server_chain = fetch_server_sent_chain(url_or_host, timeout=timeout)
    local_store = load_local_store_from_context()
    full_chain, completed = complete_chain_with_local_store(server_chain, local_store)
    return full_chain, completed

def summarize_chain(certs: Sequence[x509.Certificate]) -> List[ChainSummaryRow]:
    """
    Produce a compact, human-friendly summary of a certificate chain.
    """
    rows: List[ChainSummaryRow] = []
    for i, c in enumerate(certs):
        nbf, naf = _validity_utc(c)

        rows.append(
            ChainSummaryRow(
                index=i,
                subject_cn=_name_to_cn(c.subject),
                issuer_cn=_name_to_cn(c.issuer),
                subject_dn=_name_to_str(c.subject),
                issuer_dn=_name_to_str(c.issuer),
                not_before=nbf,   # <- tz-aware
                not_after=naf,    # <- tz-aware
                self_signed=_is_self_signed(c),
                aki_hex=_get_aki_hex(c),
                ski_hex=_get_ski_hex(c),
            )
        )
    return rows


def save_chain_pem(certs: Sequence[x509.Certificate], *, prefix: str = "chain") -> List[str]:
    """
    Save each certificate in the chain as a PEM file, prefixed and index-ordered.

    Returns:
        List of file paths written.
    """
    def _safe_cn(name: x509.Name) -> str:
        cn = _name_to_cn(name) or "no-cn"
        return "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in cn)[:80]

    paths: List[str] = []
    for i, c in enumerate(certs):
        fn = f"{prefix}-{i:02d}-{_safe_cn(c.subject)}.pem"
        with open(fn, "wb") as f:
            f.write(c.public_bytes(serialization.Encoding.PEM))
        paths.append(fn)
    return paths

def get_cert_chain(url, formatted=False):
    chain, completed = build_chain_from_url_using_local_store(url, timeout=5)

    rows = summarize_chain(chain)

    # Formatting output
    data = []
    output = []
    headers = ['index', 'commonName', 'Issuer', 'Not Before', 'Not After', 'Status']
    for r in rows:
        if formatted:
            cert = (
                f'[{r.index}]', 
                f'{r.subject_cn!s}', 
                f'{r.issuer_cn!s}',
                r.not_before.date().isoformat(),
                r.not_after.date().isoformat(),
                f"{'(self-signed)' if r.self_signed else 'Trusted'}"
            )
            data.append(cert)
        else:
            output.append((f"[{r.index}] commonName: {r.subject_cn!s} Issuer: {r.issuer_cn!s} Valid From: {r.not_before.date()} Valid To: {r.not_after.date()} Status: {'(self-signed)' if r.self_signed else 'Trusted'}"))

    if formatted:
        rows = [headers] + [(i, s, p, b, a, v) for i, s, p, b, a, v in data]
        cols = list(zip(*rows))

        COLUMN_SLACK = 2
        widths = [max(len(x)+COLUMN_SLACK for x in col) for col in cols]
        fmt = f"{{:<{widths[0]}}}{{:<{widths[1]}}}{{:<{widths[2]}}}{{:<{widths[3]}}}{{:<{widths[4]}}}{{:>{widths[5]}}}"
        output.append(fmt.format(*headers))
        output.append("-"*widths[0] + "-"*widths[1] + "-"*widths[2] + "-"*widths[3] + "-"*widths[4] + "-"*widths[5])
        for r in rows[1:]:
            output.append(fmt.format(*r))

    print(output)
    return output

def print_cert_chain(url, formatted=True):
    chain = get_cert_chain(url)
    for cert in chain:
        print({cert})

# --------------------------------------------------------------------------------------
# Optional CLI for quick manual checks
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Fetch and locally reconstruct a TLS certificate chain.")
    parser.add_argument("target", help="Hostname or URL (e.g., https://example.com)")
    parser.add_argument("--timeout", type=float, default=5.0, help="Socket timeout in seconds (default: 5.0)")
    parser.add_argument("--save-prefix", type=str, default="", help="If set, save chain PEM files with this prefix")
    parser.add_argument("--json", action="store_true", help="Print JSON summary")
    args = parser.parse_args()

    chain, completed = build_chain_from_url_using_local_store(args.target, timeout=args.timeout)

    rows = summarize_chain(chain)
    if args.json:
        # Make datetimes JSON-friendly
        def _ser(o):
            if isinstance(o, datetime):
                return o.isoformat()
            raise TypeError
        print(json.dumps({
            "target": args.target,
            "completed_locally": completed,
            "chain_len": len(chain),
            "chain": [row.__dict__ for row in rows],
        }, default=_ser, indent=2))
    else:
        print(f"Target: {args.target}")
        print(f"Completed locally: {completed}")
        print(f"Chain length: {len(chain)}")
        for r in rows:
            print(f"[{r.index}] {r.subject_cn!s}  <--  {r.issuer_cn!s}   "
                  f"{r.not_before.date()} → {r.not_after.date()} "
                  f"{'(self-signed)' if r.self_signed else ''}")

    if args.save_prefix:
        paths = save_chain_pem(chain, prefix=args.save_prefix)
        print("Saved:", ", ".join(paths))
