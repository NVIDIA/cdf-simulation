# Versioned CDF YAML configuration

`cdf.config` owns the domain-neutral versioned JobConfig envelope, loading,
plugin discovery, path handling, and command-line execution. A domain plugin
registers the concrete models and interpreters for `scene`, `appearance`,
`operations`, `annotations`, `capture`, and `postprocess`.

Import or install the selected domain plugin before validating a `JobConfig`.
The repository's PCB-AOI schema and examples are documented in
[`pcb_aoi/config/README.md`](../../pcb_aoi/config/README.md).

The generic envelope contains:

- `api_version`: the CDF authoring API version;
- `domain`: the registered domain identifier;
- `random_seed`: the deterministic run seed;
- `output_dir`: the capture output root;
- `execution`: backend execution options.

Domain section schemas are contributed through `DomainConfigAdapter`. The
generated schema includes every installed domain contribution without CDF
importing a particular domain package.
