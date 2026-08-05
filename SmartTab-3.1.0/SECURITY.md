# Security Policy

## Supported versions

Security fixes are applied to the latest minor release.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability. Use GitHub's private vulnerability reporting for this repository. Include affected versions, reproduction steps, impact, and a minimal proof of concept.

## Model bundle trust boundary

`.smarttab` bundles contain joblib/pickle payloads. Deserialization can execute code. `smarttab.load(path, trusted=True)` must only be used for a bundle created by the caller or obtained through a trusted and authenticated distribution channel.

SHA-256 bundle hashes detect accidental corruption. They do not authenticate the publisher and do not make malicious pickle content safe.

## Data handling

SmartTab processes local tabular data. Reports may contain schema names, class distributions, metrics, hardware details, and feature importance. Review report artifacts before sharing them.
