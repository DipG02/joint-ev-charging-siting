# Data directory

`processed/` contains the six consolidated CSV inputs consumed by the current Python workflow. Their schemas and relationships are documented in [../docs/INPUT_DATA_DICTIONARY.md](../docs/INPUT_DATA_DICTIONARY.md); upstream sources and release obligations are documented in [../docs/PUBLIC_DATA_SOURCES.md](../docs/PUBLIC_DATA_SOURCES.md).

These files are derived research inputs, not raw-source archives. Their source attribution and applicable upstream terms are summarized in [../DATA_LICENSE.md](../DATA_LICENSE.md). Checksums are listed in [../SHA256SUMS.txt](../SHA256SUMS.txt).

Large run outputs are deliberately excluded. In particular, the pooled segment-level sensitivity table can exceed 1 GB and should be deposited in an appropriate data repository if it is to be shared.

