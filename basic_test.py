"""Standalone smoke test for the CellWhisperer Nahual server encoder.

This script does NOT spin up a Nahual server or any IPC. It instantiates the
exact same encoder class that ``server.py`` builds (a randomly-initialized
BERT-style transcriptome encoder) and runs a forward pass on a tiny synthetic
``(N_cells, N_genes)`` matrix to verify that the model assembly and processing
logic work end-to-end inside the flake's dev shell.

Run with:
    nix develop --impure --command python basic_test.py

Expected output:
    embedding shape: (4, 256)
"""

import sys

# server.py captures sys.argv[1] at import time as the IPC address. Inject a
# placeholder so the import succeeds when this script is invoked without one.
if len(sys.argv) < 2:
    sys.argv.append("ipc:///tmp/cellwhisperer_basic_test.ipc")

import numpy  # noqa: E402

from server import setup  # noqa: E402


def main() -> None:
    processor, info = setup()
    print(f"setup info: {info}")

    numpy.random.seed(0)
    expression = numpy.random.random_sample((4, 2000)).astype(numpy.float32)

    embedding = processor(expression).cpu().numpy()
    print(f"embedding shape: {embedding.shape}")
    assert embedding.shape == (4, info["embed_dim"]), (
        f"unexpected embedding shape {embedding.shape}"
    )


if __name__ == "__main__":
    main()
