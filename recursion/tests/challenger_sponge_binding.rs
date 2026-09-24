//! Fiat-Shamir binding tests for the extension-field ([`CircuitChallenger`] `D >= 2`) sponge,
//! adapted from upstream Plonky3-recursion's `recursion/tests/challenger_sponge_binding.rs`
//! (96248cb / 7e0032d / e63b5d5) to this fork's API, in the BabyBear D=4 width-16 setting
//! zkSketch uses.
//!
//! The in-circuit challenger keeps its sponge state as `WIDTH` base-field coefficient targets.
//! Each duplex step packs those coefficients into `WIDTH / D` extension limbs, runs one
//! Poseidon2 permutation and unpacks the result back into coefficients. For the transcript to
//! bind, (a) the limbs the permutation of step `k + 1` reads must be the limbs the permutation
//! of step `k` produced, and (b) the coefficients the challenger hands out as challenges must be
//! the base-field coefficients of those limbs — both as seen by the verifier, i.e. tied together
//! on the `WitnessChecks` bus, not merely equal in an honestly generated witness.
//!
//! Each test builds the transcript circuit with the real [`CircuitChallenger`], then edits the
//! compiled [`Circuit`] in ways that leave `generate_preprocessed_columns` byte-identical
//! (asserted) and only change prover-side data: a hint executor, the witness slots a
//! permutation writes for limbs it does not publish on the bus, or the witness slots a plain
//! `recompose` row reads. Every such edit is a choice a prover assembling trace polynomials
//! directly is free to make. The resulting traces are proved against the *unedited* circuit's
//! prover data and verified, so acceptance means the verifier's constraint system does not bind
//! the value.

use std::panic::AssertUnwindSafe;

use p3_baby_bear::{BabyBear, default_babybear_poseidon2_16};
use p3_batch_stark::ProverData;
use p3_circuit::ops::{
    HintExecutor, Op, Poseidon2Config, generate_poseidon2_trace, generate_recompose_trace,
};
use p3_circuit::tables::Traces;
use p3_circuit::{Circuit, CircuitBuilder, CircuitError, WitnessId};
use p3_circuit_prover::batch_stark_prover::{poseidon2_air_builders, recompose_air_builders};
use p3_circuit_prover::common::{NpoPreprocessor, get_airs_and_degrees_with_prep};
use p3_circuit_prover::config::BabyBearConfig;
use p3_circuit_prover::{
    BatchStarkProver, CircuitProverData, ConstraintProfile, Poseidon2Preprocessor,
    RecomposePreprocessor, TablePacking, config,
};
use p3_field::extension::BinomialExtensionField;
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_poseidon2_circuit_air::BabyBearD4Width16;
use p3_recursion::challenger::CircuitChallenger;
use p3_recursion::traits::RecursiveChallenger;

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;

const D: usize = 4;
const WIDTH: usize = 16;
const RATE: usize = 8;
const CFG: Poseidon2Config = Poseidon2Config::BABY_BEAR_D4_W16;
const RATE_EXT: usize = RATE / D;
const WIDTH_EXT: usize = WIDTH / D;
/// First capacity limb: the lowest limb index the challenger permutation leaves off the bus.
const CAPACITY_LIMB: usize = RATE_EXT;

/// How `recompose_base_coeffs_to_ext` is lowered: through the dedicated recompose table
/// (zkSketch's default: `disable_recompose_npo = false`), or through the ALU `mul_add` chain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RecomposeMode {
    NpoTable,
    AluChain,
}

/// Absorb `RATE` values, then squeeze past the output buffer so a second duplex step runs.
fn build_transcript_circuit(mode: RecomposeMode) -> Circuit<EF> {
    let perm = default_babybear_poseidon2_16();
    let mut circuit = CircuitBuilder::<EF>::new();
    circuit.enable_poseidon2_perm::<BabyBearD4Width16, _>(
        generate_poseidon2_trace::<EF, BabyBearD4Width16>,
        perm,
    );
    match mode {
        RecomposeMode::NpoTable => {
            circuit.enable_recompose::<F>(generate_recompose_trace::<F, EF>);
        }
        RecomposeMode::AluChain => {
            circuit.noop_enable_recompose::<F>(generate_recompose_trace::<F, EF>);
        }
    }

    let mut challenger = CircuitChallenger::<WIDTH, RATE, Poseidon2Config>::new_babybear();
    for i in 0..RATE {
        let t = circuit.define_const(EF::from_u64(i as u64 + 1));
        RecursiveChallenger::<F, EF>::observe(&mut challenger, &mut circuit, t);
    }
    // `RATE` samples drain the output buffer; the next one forces a second permutation. Each
    // sample is consumed by an ALU op, as a verifier circuit would consume a challenge.
    for _ in 0..=RATE {
        let s = RecursiveChallenger::<F, EF>::sample(&mut challenger, &mut circuit);
        let _ = circuit.mul(s, s);
    }

    circuit.build().expect("transcript circuit builds")
}

fn is_op_type(op: &Op<EF>, needle: &str) -> bool {
    match op {
        Op::NonPrimitiveOpWithExecutor { executor, .. } => {
            format!("{:?}", executor.op_type()).contains(needle)
        }
        _ => false,
    }
}

/// Positions of the Poseidon2 permutation ops, in execution order.
fn perm_op_positions(circuit: &Circuit<EF>) -> Vec<usize> {
    (0..circuit.ops.len())
        .filter(|&i| is_op_type(&circuit.ops[i], "poseidon2_perm"))
        .collect()
}

fn npo_io(circuit: &Circuit<EF>, pos: usize) -> (Vec<Vec<WitnessId>>, Vec<Vec<WitnessId>>) {
    match &circuit.ops[pos] {
        Op::NonPrimitiveOpWithExecutor {
            inputs, outputs, ..
        } => (inputs.clone(), outputs.clone()),
        _ => panic!("op {pos} is not a non-primitive op"),
    }
}

/// Position of the `Op::Hint` that decomposes `wid` into its `D` coefficient witnesses.
fn decomposition_hint_position(circuit: &Circuit<EF>, wid: WitnessId) -> usize {
    (0..circuit.ops.len())
        .find(|&i| matches!(&circuit.ops[i], Op::Hint { inputs, .. } if inputs == &[wid]))
        .unwrap_or_else(|| panic!("no decomposition hint reading {wid:?}"))
}

fn hint_outputs(circuit: &Circuit<EF>, pos: usize) -> Vec<WitnessId> {
    match &circuit.ops[pos] {
        Op::Hint { outputs, .. } => outputs.clone(),
        _ => panic!("op {pos} is not a hint"),
    }
}

/// Position of the `recompose` row that writes `wid`.
fn recompose_position_writing(circuit: &Circuit<EF>, wid: WitnessId) -> usize {
    (0..circuit.ops.len())
        .find(|&i| match &circuit.ops[i] {
            Op::NonPrimitiveOpWithExecutor { outputs, .. } => {
                is_op_type(&circuit.ops[i], "recompose") && outputs[0][0] == wid
            }
            _ => false,
        })
        .unwrap_or_else(|| panic!("no recompose row writing {wid:?}"))
}

/// Hint that ignores its input and writes fixed base-embedded coefficients.
///
/// Stands in for a prover that fills the decomposition witnesses with values of its own
/// choosing instead of the permutation's actual output coefficients.
#[derive(Debug, Clone)]
struct ChosenCoefficients(u64);

impl HintExecutor<EF> for ChosenCoefficients {
    fn execute(
        &self,
        _inputs: &[WitnessId],
        outputs: &[WitnessId],
        witness: &mut [Option<EF>],
    ) -> Result<(), CircuitError> {
        for (i, &out) in outputs.iter().enumerate() {
            witness[out.0 as usize] = Some(EF::from_u64(self.0 + i as u64));
        }
        Ok(())
    }

    fn boxed(&self) -> Box<dyn HintExecutor<EF>> {
        Box::new(self.clone())
    }
}

/// Hint that writes the honest decomposition of its input, except that coefficient 1 is moved
/// by `-delta` and the difference is hidden in a non-base component of coefficient 0:
/// `c0' = c0 + delta * X`, `c1' = c1 - delta`, so `sum c_i' X^i == x` still holds. Every
/// coefficient is meant to be a base-field element, but `sum c_i * X^i == x` alone does not
/// say so.
#[derive(Debug, Clone)]
struct NonBaseShift(u64);

impl HintExecutor<EF> for NonBaseShift {
    fn execute(
        &self,
        inputs: &[WitnessId],
        outputs: &[WitnessId],
        witness: &mut [Option<EF>],
    ) -> Result<(), CircuitError> {
        let x = witness[inputs[0].0 as usize].expect("decomposition input is set");
        let coeffs = <EF as BasedVectorSpace<F>>::as_basis_coefficients_slice(&x).to_vec();
        let delta = F::from_u64(self.0);
        for (i, &out) in outputs.iter().enumerate() {
            let v = match i {
                0 => EF::from_basis_coefficients_fn(|j| match j {
                    0 => coeffs[0],
                    1 => delta,
                    _ => F::ZERO,
                }),
                1 => EF::from(coeffs[1] - delta),
                _ => EF::from(coeffs[i]),
            };
            witness[out.0 as usize] = Some(v);
        }
        Ok(())
    }

    fn boxed(&self) -> Box<dyn HintExecutor<EF>> {
        Box::new(self.clone())
    }
}

fn run(circuit: &Circuit<EF>) -> Traces<EF> {
    let mut runner = circuit.runner();
    runner.set_public_inputs(&[]).expect("no public inputs");
    runner.run().expect("witness generation succeeds")
}

fn witness_values(circuit: &Circuit<EF>) -> Vec<Option<EF>> {
    let mut runner = circuit.runner();
    runner.set_public_inputs(&[]).expect("no public inputs");
    runner.execute_all().expect("witness generation succeeds");
    runner.witness().to_vec()
}

/// Prove `traces` against the constraint system of `circuit`, then verify.
fn prove_and_verify(circuit: &Circuit<EF>, traces: &Traces<EF>) -> Result<(), String> {
    let table_packing = TablePacking::new(1, 1);
    let stark_config = config::baby_bear();
    let npo_preprocessors: Vec<Box<dyn NpoPreprocessor<F>>> = vec![
        Box::new(Poseidon2Preprocessor),
        // Both recompose tables: the challenger unpacks through `recompose/coeff` when the
        // recompose NPO is enabled.
        Box::new(RecomposePreprocessor::new(true)),
    ];
    let mut air_builders = poseidon2_air_builders::<BabyBearConfig, D>();
    air_builders.extend(recompose_air_builders::<BabyBearConfig, D>(1, true));

    let (airs_degrees, primitive_columns, non_primitive_columns) =
        get_airs_and_degrees_with_prep::<BabyBearConfig, EF, D>(
            circuit,
            &table_packing,
            &npo_preprocessors,
            &air_builders,
            ConstraintProfile::Standard,
        )
        .expect("preprocessed columns");
    let (airs, degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();

    let prover_data = ProverData::from_airs_and_degrees(&stark_config, &airs, &degrees);
    let circuit_prover_data =
        CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);
    let mut prover = BatchStarkProver::new(stark_config).with_table_packing(table_packing);
    prover.register_poseidon2_table::<D>(CFG);
    prover.register_recompose_table::<D>(true);

    // A bus imbalance surfaces either as a prover-side panic (the lookup debugger runs under
    // `debug_assertions`) or as a verification failure; both count as rejection.
    std::panic::catch_unwind(AssertUnwindSafe(|| {
        let proof = prover
            .prove_all_tables(traces, &circuit_prover_data)
            .map_err(|e| format!("prove: {e:?}"))?;
        prover
            .verify_all_tables(&proof)
            .map_err(|e| format!("verify: {e:?}"))
    }))
    .unwrap_or_else(|_| Err("prover panicked on an unbalanced bus".to_string()))
}

fn assert_same_constraint_system(honest: &Circuit<EF>, edited: &Circuit<EF>) {
    let honest_prep = honest
        .generate_preprocessed_columns::<D>()
        .expect("honest preprocessed columns");
    let edited_prep = edited
        .generate_preprocessed_columns::<D>()
        .expect("edited preprocessed columns");
    assert!(
        honest_prep == edited_prep,
        "the edit must leave the verifier's preprocessed columns untouched"
    );
    assert_eq!(honest.witness_count, edited.witness_count);
}

// ---------------------------------------------------------------------------
// Baseline and control
// ---------------------------------------------------------------------------

#[test]
fn honest_transcript_proves_and_verifies() {
    for mode in [RecomposeMode::NpoTable, RecomposeMode::AluChain] {
        let circuit = build_transcript_circuit(mode);
        assert!(
            perm_op_positions(&circuit).len() >= 2,
            "{mode:?}: the transcript must run at least two permutations"
        );
        let traces = run(&circuit);
        prove_and_verify(&circuit, &traces)
            .unwrap_or_else(|e| panic!("{mode:?}: honest transcript must verify: {e}"));
    }
}

/// The harness rejects a value the `WitnessChecks` bus does bind, so acceptance in the
/// tests below is evidence about the constraint system rather than about the harness.
#[test]
fn control_corrupted_const_value_is_rejected() {
    let circuit = build_transcript_circuit(RecomposeMode::NpoTable);
    let perms = perm_op_positions(&circuit);
    let (perm_inputs, _) = npo_io(&circuit, perms[0]);
    let read_const = perm_inputs[0][0];

    let mut traces = run(&circuit);
    let row = traces
        .const_trace
        .index
        .iter()
        .position(|i| *i == read_const)
        .expect("a const row feeding the first permutation");
    traces.const_trace.values[row] += EF::ONE;

    assert!(
        prove_and_verify(&circuit, &traces).is_err(),
        "a const value read over the bus must not be freely re-chosen"
    );
}

// ---------------------------------------------------------------------------
// Sponge-chain binding: the limbs permutation k + 1 reads
// ---------------------------------------------------------------------------

/// The capacity limb the second permutation reads must be the capacity limb the first
/// permutation produced.
///
/// Two circuit shapes are covered:
///
/// - The second permutation reads its capacity limb from a witness (CTL-fed capacity, the
///   shape before the capacity was chained). The edit gives the first permutation an empty
///   output slot for that limb — a slot the Poseidon2 table neither publishes on the bus nor
///   records in its trace row, so dropping the write is invisible to the verifier — and
///   replaces the decomposition hint with one that supplies coefficients of its own.
/// - The second permutation reads no capacity witness at all (capacity chained by the AIR).
///   The edit swaps the second permutation's executor for one that starts a fresh sponge
///   (zero capacity) instead of continuing the chain — an executor is prover-side code, and
///   the honest prover data still commits to the continuation row's chain selectors.
fn capacity_limb_binding(mode: RecomposeMode) {
    let honest = build_transcript_circuit(mode);
    let perms = perm_op_positions(&honest);
    let (_, first_outputs) = npo_io(&honest, perms[0]);
    let (second_inputs, _) = npo_io(&honest, perms[1]);

    let edited = if second_inputs[CAPACITY_LIMB].is_empty() {
        // Chained capacity: splice in a fresh-sponge executor for the second permutation.
        let mut donor = CircuitBuilder::<EF>::new();
        donor.enable_poseidon2_perm::<BabyBearD4Width16, _>(
            generate_poseidon2_trace::<EF, BabyBearD4Width16>,
            default_babybear_poseidon2_16(),
        );
        let limbs: Vec<_> = (0..WIDTH_EXT).map(|_| donor.public_input()).collect();
        donor
            .add_poseidon2_perm_for_challenger_chained(CFG, None, &limbs)
            .expect("donor permutation");
        let donor = donor.build().expect("donor circuit builds");
        let donor_exec = donor
            .ops
            .iter()
            .find_map(|op| match op {
                Op::NonPrimitiveOpWithExecutor { executor, .. }
                    if format!("{:?}", executor.op_type()).contains("poseidon2_perm") =>
                {
                    Some(executor.clone())
                }
                _ => None,
            })
            .expect("donor permutation op");
        let mut edited = honest.clone();
        match &mut edited.ops[perms[1]] {
            Op::NonPrimitiveOpWithExecutor { executor, .. } => *executor = donor_exec,
            _ => unreachable!(),
        }
        edited
    } else {
        let produced = first_outputs[CAPACITY_LIMB][0];
        let consumed = second_inputs[CAPACITY_LIMB][0];
        let honest_witness = witness_values(&honest);
        assert_eq!(
            honest_witness[produced.0 as usize], honest_witness[consumed.0 as usize],
            "{mode:?}: the honest witness must carry the capacity limb forward unchanged"
        );

        let mut edited = honest.clone();
        let hint_pos = decomposition_hint_position(&honest, produced);
        match &mut edited.ops[perms[0]] {
            Op::NonPrimitiveOpWithExecutor { outputs, .. } => {
                outputs[CAPACITY_LIMB].clear();
            }
            _ => unreachable!(),
        }
        match &mut edited.ops[hint_pos] {
            Op::Hint { executor, .. } => *executor = Box::new(ChosenCoefficients(1_000)),
            _ => unreachable!(),
        }
        assert_same_constraint_system(&honest, &edited);

        let edited_witness = witness_values(&edited);
        assert_ne!(
            edited_witness[consumed.0 as usize], honest_witness[produced.0 as usize],
            "{mode:?}: the edit must actually break the capacity chain"
        );
        edited
    };

    // The second permutation's rate output must actually move (it is read downstream).
    let (_, second_outputs) = npo_io(&honest, perms[1]);
    let moved = second_outputs[0][0];
    assert_ne!(
        witness_values(&edited)[moved.0 as usize],
        witness_values(&honest)[moved.0 as usize],
        "{mode:?}: the edit must change what the second permutation squeezes"
    );

    let traces = run(&edited);
    let outcome = prove_and_verify(&honest, &traces);
    println!("{mode:?} capacity edit: {outcome:?}");
    assert!(
        outcome.is_err(),
        "{mode:?}: a proof whose second permutation reads a capacity limb the first \
         permutation never produced must be rejected"
    );
}

#[test]
fn capacity_limb_is_bound_across_permutations_alu_chain() {
    capacity_limb_binding(RecomposeMode::AluChain);
}

#[test]
fn capacity_limb_is_bound_across_permutations_npo_table() {
    capacity_limb_binding(RecomposeMode::NpoTable);
}

/// The rate limb the second permutation reads must be the rate limb the first permutation
/// produced.
///
/// Under [`RecomposeMode::NpoTable`] the step-`k + 1` packing is a fresh `recompose` row that
/// creates its own witness, so the edit only has to point that row at a different, already
/// computed coefficient group: its `v_0..v_{D-1}` columns carry no bus lookup of their own.
/// When the packing is not a plain `recompose` row (ALU chain), there is nothing to re-point
/// and the limb is carried on the permutation's own output witness.
#[test]
fn rate_limb_is_bound_across_permutations_npo_table() {
    let honest = build_transcript_circuit(RecomposeMode::NpoTable);
    let perms = perm_op_positions(&honest);
    let (_, first_outputs) = npo_io(&honest, perms[0]);
    let (second_inputs, _) = npo_io(&honest, perms[1]);
    let produced = first_outputs[0][0];
    let consumed = second_inputs[0][0];
    if produced == consumed {
        // Carried on one witness id: nothing a prover could re-point.
        return;
    }

    let honest_witness = witness_values(&honest);
    assert_eq!(
        honest_witness[produced.0 as usize], honest_witness[consumed.0 as usize],
        "the honest witness must carry the rate limb forward unchanged"
    );

    let plain_recompose_writer = (0..honest.ops.len()).find(|&i| match &honest.ops[i] {
        Op::NonPrimitiveOpWithExecutor { outputs, .. } => {
            is_op_type(&honest.ops[i], "recompose") && outputs[0][0] == consumed
        }
        _ => false,
    });
    if plain_recompose_writer.is_none() {
        // Packed by the ALU chain: its operand indices are preprocessed, so re-pointing it is
        // not a free prover choice (and the coefficients it reads are bus-bound).
        assert!(
            honest
                .ops
                .iter()
                .any(|op| matches!(op, Op::Alu { out, .. } if *out == consumed)),
            "the repacked rate limb must be written by an ALU op"
        );
        return;
    }
    let target_row = recompose_position_writing(&honest, consumed);
    let donor_row = recompose_position_writing(&honest, second_inputs[1][0]);
    let (donor_inputs, _) = npo_io(&honest, donor_row);

    let mut edited = honest.clone();
    match &mut edited.ops[target_row] {
        Op::NonPrimitiveOpWithExecutor { inputs, .. } => *inputs = donor_inputs,
        _ => unreachable!(),
    }
    let same_prep = honest.generate_preprocessed_columns::<D>().unwrap()
        == edited.generate_preprocessed_columns::<D>().unwrap();
    if !same_prep {
        // The packing row advertises its coefficients: re-pointing it moves the verifier's
        // own preprocessed data, so it is not a free prover choice.
        return;
    }

    let edited_witness = witness_values(&edited);
    assert_ne!(
        edited_witness[consumed.0 as usize], honest_witness[produced.0 as usize],
        "the edit must actually break the rate chain"
    );

    let traces = run(&edited);
    assert!(
        prove_and_verify(&honest, &traces).is_err(),
        "a proof whose second permutation reads a rate limb the first permutation never \
         produced must be rejected"
    );
}

// ---------------------------------------------------------------------------
// Squeeze binding: the coefficients handed out as challenges
// ---------------------------------------------------------------------------

/// The values `sample` returns must be the base-field coefficients of the permutation's rate
/// output. The edit swaps the decomposition hint of the first permutation's first rate limb
/// for `hint`, so the challenger hands out coefficients of the prover's choosing.
fn squeezed_coefficients_binding(mode: RecomposeMode, hint: Box<dyn HintExecutor<EF>>) {
    let honest = build_transcript_circuit(mode);
    let perms = perm_op_positions(&honest);
    let (_, first_outputs) = npo_io(&honest, perms[0]);
    let limb = first_outputs[0][0];
    let hint_pos = decomposition_hint_position(&honest, limb);
    let coeff_wids = hint_outputs(&honest, hint_pos);

    let mut edited = honest.clone();
    match &mut edited.ops[hint_pos] {
        Op::Hint { executor, .. } => *executor = hint,
        _ => unreachable!(),
    }
    assert_same_constraint_system(&honest, &edited);

    let honest_witness = witness_values(&honest);
    let mut runner = edited.runner();
    runner.set_public_inputs(&[]).unwrap();
    let edited_witness = match runner.execute_all() {
        Ok(_) => runner.witness().to_vec(),
        // Witness generation itself refuses the edit: the coefficients are bound.
        Err(_) => return,
    };
    assert!(
        coeff_wids
            .iter()
            .any(|w| edited_witness[w.0 as usize] != honest_witness[w.0 as usize]),
        "{mode:?}: the edit must actually move a squeezed coefficient"
    );
    assert_eq!(
        edited_witness[limb.0 as usize], honest_witness[limb.0 as usize],
        "{mode:?}: the permutation output itself is untouched"
    );

    let traces = run(&edited);
    assert!(
        prove_and_verify(&honest, &traces).is_err(),
        "{mode:?}: a proof whose squeezed challenges are not the coefficients of the \
         permutation output must be rejected"
    );
}

#[test]
fn squeezed_coefficients_are_bound_npo_table_chosen() {
    squeezed_coefficients_binding(
        RecomposeMode::NpoTable,
        Box::new(ChosenCoefficients(1_000)),
    );
}

#[test]
fn squeezed_coefficients_are_bound_alu_chain_chosen() {
    squeezed_coefficients_binding(
        RecomposeMode::AluChain,
        Box::new(ChosenCoefficients(1_000)),
    );
}

#[test]
fn squeezed_coefficients_are_bound_npo_table_non_base_shift() {
    squeezed_coefficients_binding(RecomposeMode::NpoTable, Box::new(NonBaseShift(7)));
}

/// Without `enable_recompose` there is no `recompose/coeff` table to unpack through, so the
/// challenger falls back to the ALU chain and this gap stays open for such circuits (zkSketch's
/// recursion enables the recompose NPO; upstream f89d776 + 1ba9121 instead refuse to build).
#[test]
#[ignore = "open soundness gap without enable_recompose: the ALU chain leaves D-1 free base dims per coefficient"]
fn squeezed_coefficients_are_bound_alu_chain_non_base_shift() {
    squeezed_coefficients_binding(RecomposeMode::AluChain, Box::new(NonBaseShift(7)));
}

// ---------------------------------------------------------------------------
// Extension challenge binding: `sample_ext`
// ---------------------------------------------------------------------------

/// Absorb `RATE` values and draw one extension challenge with `sample_ext`, which an ALU op
/// then consumes (as a verifier circuit would).
fn build_sample_ext_circuit(mode: RecomposeMode) -> (Circuit<EF>, WitnessId) {
    let perm = default_babybear_poseidon2_16();
    let mut circuit = CircuitBuilder::<EF>::new();
    circuit.enable_poseidon2_perm::<BabyBearD4Width16, _>(
        generate_poseidon2_trace::<EF, BabyBearD4Width16>,
        perm,
    );
    match mode {
        RecomposeMode::NpoTable => {
            circuit.enable_recompose::<F>(generate_recompose_trace::<F, EF>);
        }
        RecomposeMode::AluChain => {
            circuit.noop_enable_recompose::<F>(generate_recompose_trace::<F, EF>);
        }
    }
    let mut challenger = CircuitChallenger::<WIDTH, RATE, Poseidon2Config>::new_babybear();
    for i in 0..RATE {
        let t = circuit.define_const(EF::from_u64(i as u64 + 1));
        RecursiveChallenger::<F, EF>::observe(&mut challenger, &mut circuit, t);
    }
    let alpha = RecursiveChallenger::<F, EF>::sample_ext(&mut challenger, &mut circuit);
    let _ = circuit.mul(alpha, alpha);
    let circuit = circuit.build().expect("sample_ext circuit builds");
    let alpha_wid = circuit.expr_to_widx[&alpha];
    (circuit, alpha_wid)
}

/// The extension challenge `sample_ext` returns must be the recomposition of the squeezed
/// base coefficients. Under [`RecomposeMode::NpoTable`] that recomposition is a plain
/// `recompose` row whose coefficient inputs are not on the bus, so the edit re-points it at the
/// same squeezed coefficients in a different order: the preprocessed columns do not change and
/// the proof carries a challenge of the prover's choosing. When no plain `recompose` row writes
/// the challenge there is nothing to re-point.
#[test]
fn sampled_ext_challenge_is_bound_npo_table() {
    let (honest, alpha_wid) = build_sample_ext_circuit(RecomposeMode::NpoTable);
    let Some(pos) = (0..honest.ops.len()).find(|&i| match &honest.ops[i] {
        Op::NonPrimitiveOpWithExecutor { outputs, .. } => {
            is_op_type(&honest.ops[i], "recompose") && outputs[0][0] == alpha_wid
        }
        _ => false,
    }) else {
        return;
    };
    let (inputs, _) = npo_io(&honest, pos);
    let mut edited = honest.clone();
    match &mut edited.ops[pos] {
        Op::NonPrimitiveOpWithExecutor { inputs: ins, .. } => {
            let mut rev = inputs[0].clone();
            rev.reverse();
            *ins = vec![rev];
        }
        _ => unreachable!(),
    }
    let same_prep = honest.generate_preprocessed_columns::<D>().unwrap()
        == edited.generate_preprocessed_columns::<D>().unwrap();
    if !same_prep {
        return;
    }

    let honest_alpha = witness_values(&honest)[alpha_wid.0 as usize].expect("alpha computed");
    let forged_alpha = witness_values(&edited)[alpha_wid.0 as usize].expect("alpha computed");
    assert_ne!(honest_alpha, forged_alpha, "the edit must move the challenge");

    let honest_traces = run(&honest);
    prove_and_verify(&honest, &honest_traces).expect("honest sample_ext proof verifies");

    let forged_traces = run(&edited);
    assert!(
        prove_and_verify(&honest, &forged_traces).is_err(),
        "a proof asserting an extension challenge that is not the recomposition of the \
         squeezed coefficients must be rejected"
    );
}

/// Under [`RecomposeMode::AluChain`] the step-`k + 1` packing of a rate limb is deduplicated
/// back onto the permutation's own output witness; a capacity limb is either carried the same
/// way or (once chained in the AIR) not read from a witness at all.
#[test]
fn alu_chain_repacking_reuses_the_permutation_output_witness() {
    let circuit = build_transcript_circuit(RecomposeMode::AluChain);
    let perms = perm_op_positions(&circuit);
    let (_, first_outputs) = npo_io(&circuit, perms[0]);
    let (second_inputs, _) = npo_io(&circuit, perms[1]);
    for limb in 0..WIDTH_EXT {
        if limb >= RATE_EXT && second_inputs[limb].is_empty() {
            continue;
        }
        assert_eq!(
            first_outputs[limb][0], second_inputs[limb][0],
            "limb {limb} must be carried on one witness id, not repacked onto a new one"
        );
    }
}

/// Every `D >= 2` challenger permutation after the first continues the sponge chain: it reads
/// no capacity witness, so its capacity can only come from the previous row's output.
#[test]
fn continuation_permutations_read_no_capacity_witness() {
    for mode in [RecomposeMode::NpoTable, RecomposeMode::AluChain] {
        let circuit = build_transcript_circuit(mode);
        let perms = perm_op_positions(&circuit);
        let (first_inputs, _) = npo_io(&circuit, perms[0]);
        assert!(
            (0..WIDTH_EXT).all(|l| !first_inputs[l].is_empty()),
            "{mode:?}: the chain start reads its whole state (zero capacity) over the bus"
        );
        for &p in &perms[1..] {
            let (inputs, _) = npo_io(&circuit, p);
            assert!(
                (RATE_EXT..WIDTH_EXT).all(|l| inputs[l].is_empty()),
                "{mode:?}: a continuation permutation must not read its capacity from a witness"
            );
        }
    }
}

/// `clear` opens a fresh sponge chain: the first permutation after it is a chain start that
/// reads its whole state (the zero capacity included) over the bus, like the transcript's very
/// first permutation, and the resulting mid-table chain start proves and verifies. A cleared
/// challenger that kept its last permutation would instead continue the pre-clear chain and
/// read no capacity witness.
#[test]
fn clear_starts_a_fresh_sponge_chain() {
    for mode in [RecomposeMode::NpoTable, RecomposeMode::AluChain] {
        let perm = default_babybear_poseidon2_16();
        let mut circuit = CircuitBuilder::<EF>::new();
        circuit.enable_poseidon2_perm::<BabyBearD4Width16, _>(
            generate_poseidon2_trace::<EF, BabyBearD4Width16>,
            perm,
        );
        match mode {
            RecomposeMode::NpoTable => {
                circuit.enable_recompose::<F>(generate_recompose_trace::<F, EF>);
            }
            RecomposeMode::AluChain => {
                circuit.noop_enable_recompose::<F>(generate_recompose_trace::<F, EF>);
            }
        }

        let mut challenger = CircuitChallenger::<WIDTH, RATE, Poseidon2Config>::new_babybear();
        for round in 0..2u64 {
            for i in 0..RATE {
                let t = circuit.define_const(EF::from_u64(round * 100 + i as u64 + 1));
                RecursiveChallenger::<F, EF>::observe(&mut challenger, &mut circuit, t);
            }
            for _ in 0..=RATE {
                let s = RecursiveChallenger::<F, EF>::sample(&mut challenger, &mut circuit);
                let _ = circuit.mul(s, s);
            }
            if round == 0 {
                RecursiveChallenger::<F, EF>::clear(&mut challenger, &mut circuit);
            }
        }
        let circuit = circuit.build().expect("transcript circuit builds");

        let perms = perm_op_positions(&circuit);
        assert_eq!(
            perms.len(),
            4,
            "{mode:?}: two permutations before and two after the clear"
        );
        for (k, &p) in perms.iter().enumerate() {
            let (inputs, _) = npo_io(&circuit, p);
            let chain_start = k % 2 == 0;
            assert_eq!(
                (RATE_EXT..WIDTH_EXT).all(|l| !inputs[l].is_empty()),
                chain_start,
                "{mode:?}: permutation {k} must {} its capacity over the bus",
                if chain_start { "read" } else { "not read" }
            );
        }

        let traces = run(&circuit);
        prove_and_verify(&circuit, &traces)
            .unwrap_or_else(|e| panic!("{mode:?}: honest cleared transcript must verify: {e}"));
    }
}
