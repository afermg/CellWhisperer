"""Nahual server for CellWhisperer.

CellWhisperer is a multimodal model bridging scRNA-seq (transcriptomics) and
natural language. Its primary input is a gene-expression matrix (cells x genes)
and its primary output (used here) is a per-cell embedding from the
transcriptome encoder. The natural-language head is intentionally NOT exposed
through this server.

Important caveats
-----------------
The full upstream model is tightly coupled to scRNA-seq pipelines:
  * geneformer / scgpt / uce backends (extra runtime deps and gene tokenizers)
  * anndata + scanpy + biomart for ENSEMBL gene mapping
  * pretrained Lightning checkpoints (`cellwhisperer_clip_v1.ckpt`)
  * a per-cell tokenization step (`TranscriptomeTextDualEncoderProcessor`)

Inside Nix (and without the model checkpoints / gene-symbol fixtures), we
cannot load the real CellWhisperer weights. Following Nahual's "scaffold a
minimal-but-runnable encoder" fallback, this server constructs a small
BERT-based transcriptome encoder with random init that mirrors the
Geneformer-style architecture used by the real model (a `BertModel` whose
pooled output is used as the cell embedding). Real scRNA workflows would
replace `setup` with the upstream `load_cellwhisperer_model(...)` call and
replace `process` with the proper tokenization + forward.

Run with:
    nix run --impure . -- ipc:///tmp/cellwhisperer.ipc
or:
    python server.py ipc:///tmp/cellwhisperer.ipc
"""

import sys
from functools import partial
from typing import Callable

import numpy
import pynng
import torch
import trio
from nahual.server import responder
from transformers import BertConfig, BertModel

address = sys.argv[1]


def setup(
    hidden_size: int = 256,
    num_hidden_layers: int = 4,
    num_attention_heads: int = 4,
    intermediate_size: int = 512,
    max_position_embeddings: int = 4096,
    vocab_size: int = 25426,
    device: int | None = None,
    weights: str | None = None,
) -> tuple[Callable, dict]:
    """Build a small BERT-style transcriptome encoder for CellWhisperer.

    The real CellWhisperer transcriptome head is a Geneformer (BERT) on top of
    a gene-rank tokenization. Here we scaffold a randomly-initialized BERT of
    similar shape so the server is bootable without the upstream checkpoint.

    Parameters
    ----------
    hidden_size : int
        Transformer hidden size; this is also the returned embedding width.
    num_hidden_layers, num_attention_heads, intermediate_size : int
        Standard BERT shape parameters.
    max_position_embeddings : int
        Max number of "gene tokens" per cell. Inputs longer than this are
        truncated; shorter inputs are padded.
    vocab_size : int
        Size of the gene-token vocab (Geneformer uses 25,426).
    device : int | None
        CUDA device index. None -> cuda:0 if available, else cpu.
    weights : str | None
        Optional path to a state-dict to load into the BERT. None -> random.
    """
    if device is None:
        device = 0
    if torch.cuda.is_available():
        torch_device = torch.device(int(device))
    else:
        torch_device = torch.device("cpu")

    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
        pad_token_id=0,
    )
    model = BertModel(config)

    if weights is not None:
        import os

        if os.path.exists(weights):
            state_dict = torch.load(weights, map_location="cpu")
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            # Strip common prefixes that Lightning / dual-encoder checkpoints
            # tend to use.
            state_dict = {
                k.replace("model.transcriptome_model.geneformer_model.bert.", "")
                .replace("transcriptome_model.geneformer_model.bert.", "")
                .replace("module.", ""): v
                for k, v in state_dict.items()
            }
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            load_info = {"missing": len(missing), "unexpected": len(unexpected)}
        else:
            load_info = {"missing": 0, "unexpected": 0, "weights": "missing-path"}
    else:
        load_info = {"missing": 0, "unexpected": 0, "weights": "random"}

    model.to(torch_device).eval()

    info = {
        "device": str(torch_device),
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "max_position_embeddings": max_position_embeddings,
        "vocab_size": vocab_size,
        "embed_dim": hidden_size,
        "load": load_info,
        "note": (
            "Scaffold encoder with random init; upstream CellWhisperer "
            "checkpoint not loaded (would require Geneformer/anndata fixtures)."
        ),
    }
    processor = partial(
        process,
        model=model,
        device=torch_device,
        max_position_embeddings=max_position_embeddings,
        vocab_size=vocab_size,
    )
    return processor, info


def process(
    expression: numpy.ndarray,
    model: BertModel,
    device: torch.device,
    max_position_embeddings: int,
    vocab_size: int,
) -> torch.Tensor:
    """Forward a (cells x genes) expression matrix through the encoder.

    Parameters
    ----------
    expression : numpy.ndarray
        2D array shaped ``(N_cells, N_genes)`` of (typically) raw or
        log-normalized counts. Real CellWhisperer expects a gene-rank
        tokenization upstream; here we approximate it by using each cell's
        top-expressed genes (by index) as token IDs ranked by expression.
        This produces a plausible (but smoke-test-only) embedding.

    Returns
    -------
    torch.Tensor of shape (N_cells, embed_dim)
        Pooled cell embeddings. Nahual's responder will move this to numpy.
    """
    if expression.ndim != 2:
        raise ValueError(
            f"Expected 2D (cells, genes) array; got shape {expression.shape}"
        )

    n_cells, n_genes = expression.shape

    # Rank genes by descending expression per cell, take up to max_position_embeddings.
    # Token IDs are gene indices clamped into the vocab range. The result mimics
    # Geneformer-style "rank value encoding": the model sees a sequence of gene
    # tokens ordered from most to least expressed.
    keep = min(max_position_embeddings, n_genes)
    # argsort ascending; reverse for descending. Use stable to keep determinism.
    order = numpy.argsort(-expression, axis=1, kind="stable")[:, :keep]
    # Map gene indices into the vocab. Reserve token 0 for [PAD], shift by 1
    # and clamp.
    token_ids = (order % (vocab_size - 1)).astype(numpy.int64) + 1
    attention_mask = numpy.ones_like(token_ids, dtype=numpy.int64)

    input_ids_t = torch.from_numpy(token_ids).to(device)
    attn_t = torch.from_numpy(attention_mask).to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids_t, attention_mask=attn_t)
        # pooler_output -> (N, hidden_size). Use it as the cell embedding.
        cell_embeds = outputs.pooler_output

    return cell_embeds


async def main():
    with pynng.Rep0(listen=address, recv_timeout=300) as sock:
        print(f"CellWhisperer server listening on {address}", flush=True)
        async with trio.open_nursery() as nursery:
            responder_curried = partial(responder, setup=setup)
            nursery.start_soon(responder_curried, sock)


if __name__ == "__main__":
    try:
        trio.run(main)
    except KeyboardInterrupt:
        pass
