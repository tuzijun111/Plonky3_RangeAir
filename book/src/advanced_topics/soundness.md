# Soundness and Security

This section describes the security model, current guarantees, and known limitations.

## What is proven

A recursive proof attests that:

1. The original computation (base layer) was executed correctly according to the AIR constraints.
2. Each intermediate STARK verification was performed correctly: commitments were checked, FRI queries were sampled from the transcript, Merkle paths were verified, and polynomial evaluations matched.
3. The Fiat-Shamir transcript was computed consistently — the circuit challenger produces identical challenges to the native Plonky3 challenger given the same observations.

## Security parameters

FRI soundness depends on several parameters working together. However, it is not generally correct to
model security as “`num_queries × log_blowup` bits”. That heuristic relied on strong proximity-gap / 
correlated-agreement assumptions that are no longer believed to hold in full generality.
See for instance:

- https://eprint.iacr.org/2025/2010
- https://eprint.iacr.org/2025/2046

Instead, the soundness error must be derived from a *proven bound* for the specific FRI variant
and parameter regime being used.

| Parameter | Role |
|-----------|------|
| `log_blowup` | Sets the Reed–Solomon code rate (blowup factor). This affects the code distance and proximity gap, but does **not** directly translate into a fixed number of “bits per query”. |
| `num_queries` | Number of independent FRI consistency checks. Increasing this reduces soundness error according to the *actual* FRI soundness bound being used. |
| `query_pow_bits` | Proof-of-work grinding. Adds `query_pow_bits` bits of security independently of the code-theoretic soundness term. |

Let:

    ε_FRI = soundness error derived from the relevant FRI theorem/bound
            (depends on blowup, proximity parameter δ, domain size,
             field size, and list-decodability/proximity-gap behavior)

Then the security level should be expressed as:

    security ≈ -log2(ε_FRI) + query_pow_bits

Crucially, `-log2(ε_FRI)` must be computed from a proven soundness bound for the specific FRI configuration.
It should not be replaced by `num_queries × log_blowup` unless an additional (explicitly stated)
conjectural assumption is being made.

## Cryptographic components verified in-circuit

### Merkle tree verification

Every MMCS opening proof (Merkle path) is verified in-circuit via Poseidon2 hashing. The circuit:
- Hashes sibling pairs up the tree
- Checks that the reconstructed root matches the committed root (a public input)
- Handles position-dependent ordering (left vs right sibling)

### FRI verification

The circuit performs the full FRI verification protocol:
- Samples folding challenges (beta) from the transcript
- Samples query indices from the transcript
- Verifies proof-of-work witnesses
- Checks the fold chain: at each FRI round, verifies that the folded polynomial evaluations are consistent with the committed Merkle trees
- Evaluates and checks the final polynomial

### Fiat-Shamir challenger

The circuit challenger implements a duplex sponge construction identical to Plonky3's native `DuplexChallenger`. It absorbs commitments and opened values in the same order as the native verifier, producing identical challenges. See [Hashing and Fiat-Shamir](./hashing.md) for details on transcript compatibility.

### ZK mode

The recursive verifier supports ZK-enabled STARK configurations, including batch recursion transcript/randomization handling.

With a ZK PCS, the Plonky3 core this fork builds on (the `zk-logup-mask` branch) blinds the
global LogUp cumulative sums, which a proof reveals in the clear and which would otherwise be a
deterministic function of the witness and of challenges the verifier can recompute:

- Each global lookup (one per `push_interaction`) gets an extension-field mask `rho`, stored as
  `D` constant main columns appended after the AIR's own `width()` columns. The masks of each bus
  are drawn from the PCS hiding randomness and sum to zero over the batch.
- The mask columns are committed with the main trace, i.e. before the lookup challenges are
  sampled, so they are constants fixed in advance and cannot balance an unbalanced bus. Only the
  last-row LogUp binding reads them: `(c' - rho - s[n-1]) * den - num = 0`, and the revealed value
  is `c' = rho + (true sum)`. The per-bus check that the revealed values sum to zero is unchanged.
- The lookup mode is fixed by the committed main width. The native verifier
  (`BatchStarkProver::verify_all_tables`, which rebuilds the lookups from AIRs it reconstructs
  itself, then the core `verify_batch`) and the in-circuit verifier
  (`verify_p3_batch_proof_circuit`, which rebuilds them the same way before
  `verify_batch_circuit`) derive the expected width as
  `width() + lookup_mask_width(lookups, D, SC::Pcs::ZK)` from those lookups and their own
  configuration, never from the proof, check the opened width against it, and bind it in the
  transcript. The prover-side transcript replay `generate_batch_challenges` only binds the width
  it derives from the lookups it is given (`common_data.lookups`); it does not check opened
  widths. Without ZK the width, the constraints and the proofs are unchanged.

Requirements:

- **AIR contract.** Under ZK an AIR with global lookups sees trailing mask columns in its main
  window. It must only read its own `width()` columns, must not assert on the window length and
  must not let its constraint set depend on it (the ALU and the Poseidon1/Poseidon2 circuit AIRs
  follow this). `recursion/tests/zk_mask_air_contract.rs` checks the Const, Public, ALU (with
  packed Horner), Poseidon2, recompose, Permute, Range (D=1 `add_range` and D=4
  `add_range_bits`), Expose and Poseidon1 tables, symbolically and, in a debug build, by
  proving masked traces. Other tables (for example a new NPO) are not covered until added there.
- **Hiding budget.** A column is hidden only if its table is taller than the number of values the
  verifier learns about it (openings at `zeta` and `g * zeta`, one value per FRI query). Under ZK
  `BatchStarkProver` refuses a `TablePacking` whose `min_trace_height` is below
  `zk_min_trace_height(challenge degree)`: the budget
  `zk_hiding_budget(num_queries, challenge degree, 2) = next_pow2(2 * (num_queries + 2 * degree))`
  (128 for 54 queries and degree 4), or a larger floor set with `with_zk_min_trace_height`
  (which can only raise it). The PCS does not expose its query count, so the count is declared
  with `with_zk_num_queries` (default `DEFAULT_ZK_NUM_QUERIES` = 54); a caller with more queries
  must declare them, and declaring fewer than the PCS uses weakens hiding. The default packing
  (`min_trace_height` 1, also used by `ProveNextLayerParams::default`) is non-ZK only.
- **Randomness.** The PCS RNG must be a CSPRNG that is not replayed across proofs. Seeded test
  RNGs replay the masks, and so do clones of a PCS whose RNG copies its state on clone:
  `prove_next_layer` and `prove_aggregation_layer` without a prep cache prove with a fresh
  `config.clone()` on every call, so under ZK their config's RNG must reseed on clone.

Tests: `recursion/tests/zk_logup_mask.rs` (WitnessChecks only) and
`recursion/tests/zk_logup_mask_npo.rs` (PermuteBus, RangeBus and WitnessChecks, with Permute,
Range, Expose and Poseidon1 tables) verify honest masked proofs natively and in circuit, and
reject forgeries made by the prover through the test-only mask hook
(`BatchStarkProver::prove_all_tables_with_mask_hook`, feature `test-hooks`), whose transcripts
are consistent with what they reveal: a same-bus shift of the revealed sums or a mask offset
cancelled in the revealed sum (caught by the last-row binding), and masks that do not sum to zero
on a bus or that cancel across two buses (caught by the per-bus check). A zero-sum
re-randomization of one bus's masks is accepted.

Still revealed under ZK: table heights, public values, and the masked sums themselves (uniform
subject to summing to zero per bus).

### Challenger Poseidon2: CTL-verified

The Fiat-Shamir challenger's Poseidon2 permutations are connected to the Poseidon2 AIR table via cross-table lookups (CTLs). The circuit builder's `add_poseidon2_perm_for_challenger` / `add_poseidon2_perm_for_challenger_base` use the standard Poseidon2 non-primitive op with full input and rate-output CTL exposure; the executor runs the real permutation and the lookup argument enforces that the (input, output) pair appears in the Poseidon2 table. The MMCS Poseidon2 calls (Merkle verification) are also CTL-verified.

## Current limitations

### Fixed Poseidon2 parameters

The recursion stack currently requires `WIDTH = 16` and `RATE = 8` for 32-bit fields with degree-4 extensions. Future versions will support configurable parameters.

## Verifier trust model

| Component | How it's verified |
|-----------|------------------|
| Commitment openings | Merkle path verification (Poseidon2, CTL-enforced) |
| FRI fold chain | Algebraic consistency checks in-circuit |
| FRI query indices | Sampled in-circuit from transcript |
| Proof-of-work | Verified in-circuit |
| Fiat-Shamir challenges | Circuit challenger (CTL-verified against Poseidon2 AIR) |
| AIR constraint satisfaction | Evaluated in-circuit via symbolic-to-circuit translation |
| Lookup argument | LogUp verification in-circuit |
