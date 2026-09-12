# Data licensing and attribution

Dipteran's **code** is MIT licensed (see [LICENSE](LICENSE)). The **data** it
consumes is not, and the two are independent. Nothing in this document grants
you rights to the datasets below; it records the terms this project operates
under and how to attribute the work.

## FlyWire / FAFB connectome

Dipteran downloads, but never redistributes, the FlyWire v783 whole-brain
connectome of *Drosophila melanogaster*.

**Treat it as CC BY-NC 4.0** — attribution required, non-commercial use only.

Upstream sources conflict on this point: the Zenodo record's metadata says
CC BY 4.0, while <https://flywire.ai/guidelines> states CC BY-NC 4.0. Where
two statements of the same licence disagree, this project takes the more
restrictive one as binding. If you intend commercial use, resolve the
discrepancy with the FlyWire consortium directly rather than relying on the
Zenodo field.

Practical consequences, enforced by [.gitignore](.gitignore):

- The raw connectome is **never** committed to this repository.
- Neither is any artefact compiled from it — adjacency matrices, neuron index
  arrays, resolved cluster tables. A derived matrix is still the data.
- Every user fetches it themselves, under these terms, into the gitignored
  `data/` directory.

### What gets fetched

| File | Source | Size |
|---|---|---|
| `proofread_connections_783.feather` | [Zenodo record 10676866](https://zenodo.org/records/10676866) | 852 MB |
| `Supplemental_file1_neuron_annotations.tsv` | [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations) | ~25 MB |

Both are openly downloadable — no account gates the bulk data. A
FlyWire/CAVE account is still worth having for interactive annotation lookup
via [Codex](https://codex.flywire.ai/), but it is not a prerequisite for
running this project.

### Required citations

> Dorkenwald, S., Matsliah, A., Sterling, A.R., Schlegel, P., Yu, S.-c.,
> McKellar, C.E., *et al.* (2024). Neuronal wiring diagram of an adult brain.
> *Nature* 634, 124–138.

> Schlegel, P., Yin, Y., Bates, A.S., Dorkenwald, S., Eichler, K., Brooks, P.,
> *et al.* (2024). Whole-brain annotation and multi-connectome cell typing of
> *Drosophila*. *Nature* 634, 139–152.

The underlying electron-microscopy volume (FAFB):

> Zheng, Z., Lauritzen, J.S., Perlman, E., Robinson, C.G., Nichols, M.,
> Milkie, D., *et al.* (2018). A complete electron microscopy volume of the
> brain of adult *Drosophila melanogaster*. *Cell* 174, 730–743.

## OpenStreetMap

Road network geometry comes from OpenStreetMap via the Overpass API, and is
licensed under the [Open Database License
(ODbL) 1.0](https://opendatacommons.org/licenses/odbl/). © OpenStreetMap
contributors.

Derived OSM artefacts (`cache/*.osm.xml`, `*.graphml`, `*.net.xml`) are also
gitignored — not for licensing reasons, but because they are large and
reproducible from `config.yaml`.

## Map tiles

The frontend renders [OpenFreeMap](https://openfreemap.org/) tiles, which are
free to use without an API key and are themselves built from OpenStreetMap
data.

## SUMO

[Eclipse SUMO](https://eclipse.dev/sumo/) is used as an external tool, not
vendored. It is licensed under EPL-2.0.
