# CDF API changelog

The CDF authoring API follows semantic versioning:

- **Major** versions contain incompatible changes and require migration.
- **Minor** versions add backward-compatible fields or accepted values.
- **Patch** versions clarify documentation or fix implementation behavior while
  preserving the set of structurally valid configurations.

[`VERSION.md`](VERSION.md) is the authoritative source for the current API
version. The headings below are the historical release record. YAML JobConfig
documents use `api_version` to select this contract; their schema evolves with
the API version.

The domain-neutral envelope and execution models live in `cdf/config`.
Registered domain plugins provide the `scene`, `appearance`, `operations`,
`annotations`, `capture`, and `postprocess` section models and interpreters.
The generated JSON Schema and public CDF Python API define the versioned
authoring contract.

## 1.0.0 — Initial release

Initial public release of the versioned CDF Python authoring API and equivalent
YAML JobConfig representation.
