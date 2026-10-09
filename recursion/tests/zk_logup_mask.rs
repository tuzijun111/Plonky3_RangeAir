//! Zero-knowledge LogUp sum masks through the circuit prover and the in-circuit verifier.
//!
//! With a ZK PCS the core batch prover appends `D` main columns per global lookup holding an
//! extension-field mask `rho`, commits them with the main trace (before the lookup challenges),
//! draws the masks of each bus so that they sum to zero over the batch, and reveals
//! `rho + (true cumulative sum)`. The last-row LogUp binding reads the mask.
//!
//! These tests cover the fork side: the in-circuit verifier derives the masked width from its
//! own lookups and PCS type, accepts honest masked proofs (also with Poseidon2, recompose and
//! packed-Horner ALU tables, over a recursion chain), rejects sum-preserving tampering (after
//! the fact, and transcript-consistent through the test-only mask hook) and stripped mask
//! columns, and builds a circuit that does not depend on the masks. All global lookups here are
//! on the WitnessChecks bus; `zk_logup_mask_npo.rs` covers the NPO buses.

mod common;

use std::collections::BTreeMap;

use p3_air::BaseAir;
use p3_batch_stark::ProverData;
use p3_circuit::ops::{generate_poseidon2_trace, generate_recompose_trace};
use p3_circuit::{Circuit, CircuitBuilder, NonPrimitiveOpId, Traces};
use p3_circuit_prover::batch_stark_prover::{
    BatchStarkProof, poseidon2_air_builders, recompose_air_builders,
};
use p3_circuit_prover::common::{NpoPreprocessor, get_airs_and_degrees_with_prep};
use p3_circuit_prover::{
    BatchStarkProver, CircuitProverData, ConstraintProfile, DEFAULT_ZK_MIN_TRACE_HEIGHT,
    LookupMaskPlan, Poseidon2Preprocessor, Poseidon2Prover, RecomposePreprocessor, TablePacking,
    TableProver, recompose_table_provers,
};
use p3_fri::{FriParameters, HidingFriPcs};
use p3_lookup::logup::LogUpGadget;
use p3_lookup::{lookup_mask_width, num_global_lookups};
use p3_poseidon2_circuit_air::KoalaBearD4Width16;
use p3_recursion::pcs::fri::{
    FriVerifierParams, HidingFriProofTargets, InputProofTargets, MerkleCapTargets,
    RecExtensionValMmcs, RecValMmcs, Witness,
};
use p3_recursion::pcs::set_fri_mmcs_private_data;
use p3_recursion::verifier::verify_p3_batch_proof_circuit;
use p3_recursion::{BatchStarkVerifierInputsBuilder, Poseidon2Config, VerificationError};
use p3_test_utils::koala_bear_params::*;
use p3_test_utils::test_fri_scalars;
use p3_uni_stark::StarkGenericConfig;
use rand::SeedableRng;
use rand::rngs::SmallRng;

use crate::common::InnerFriGeneric;

type InnerFri = InnerFriGeneric<MyConfig, MyHash, MyCompress, DIGEST_ELEMS>;

/// Test-only ZK config. The hiding RNG is a seeded `SmallRng`: each proof below uses its own
/// seed, so masks differ across proofs. Seeded RNGs must never be used for real proofs.
type MyPcsZk = HidingFriPcs<F, Dft, MyMmcs, ChallengeMmcs, SmallRng>;
type MyConfigZk = StarkConfig<MyPcsZk, Challenge, Challenger>;
type InnerFriZk = HidingFriProofTargets<
    F,
    Challenge,
    RecExtensionValMmcs<
        F,
        Challenge,
        DIGEST_ELEMS,
        RecValMmcs<F, DIGEST_ELEMS, MyHash, MyCompress>,
    >,
    InputProofTargets<F, Challenge, RecValMmcs<F, DIGEST_ELEMS, MyHash, MyCompress>>,
    Witness<F>,
>;
type ValInputProof =
    InputProofTargets<F, Challenge, RecValMmcs<F, DIGEST_ELEMS, MyHash, MyCompress>>;

const POSEIDON2: Poseidon2Config = Poseidon2Config::KOALA_BEAR_D4_W16;

fn zk_config(seed: u64) -> MyConfigZk {
    let perm = default_koalabear_poseidon2_16();
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = MyMmcs::new(hash, compress, 0);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let fri_params = FriParameters::new_testing(challenge_mmcs, 0);
    let pcs = MyPcsZk::new(
        Dft::default(),
        val_mmcs,
        fri_params,
        2,
        SmallRng::seed_from_u64(seed),
    );
    MyConfigZk::new(pcs, Challenger::new(perm))
}

/// Full in-circuit FRI verification (Merkle openings included) for the test FRI parameters.
fn fri_verifier_params() -> FriVerifierParams {
    let s = test_fri_scalars();
    FriVerifierParams::with_mmcs(
        s.log_blowup,
        s.log_final_poly_len,
        s.commit_pow_bits,
        s.query_pow_bits,
        s.num_queries,
        POSEIDON2,
    )
}

/// Packing for every ZK proof here: tables padded to the default hiding budget.
fn zk_packing(public_lanes: usize, alu_lanes: usize) -> TablePacking {
    TablePacking::new(public_lanes, alu_lanes).with_min_trace_height(DEFAULT_ZK_MIN_TRACE_HEIGHT)
}

/// A proof together with what is needed to verify it, and the AIR widths of its tables.
struct Proved<SC: StarkGenericConfig + 'static> {
    proof: BatchStarkProof<SC>,
    cpd: CircuitProverData<SC>,
    air_widths: Vec<usize>,
}

fn fib_circuit(n: usize) -> (Circuit<F>, F) {
    let mut builder = CircuitBuilder::new();
    let expected_result = builder.alloc_public_input("expected_result");
    let mut a = builder.alloc_const(F::ZERO, "F(0)");
    let mut b = builder.alloc_const(F::ONE, "F(1)");
    for _ in 2..=n {
        let next = builder.add(a, b);
        a = b;
        b = next;
    }
    builder.connect(b, expected_result);
    let mut x = (F::ZERO, F::ONE);
    for _ in 2..=n {
        x = (x.1, x.0 + x.1);
    }
    (builder.build().unwrap(), x.1)
}

/// Fibonacci circuit (Const/Public/ALU tables, all on the WitnessChecks bus) proved with ZK.
fn prove_fib_zk(seed: u64) -> Proved<MyConfigZk> {
    let p = prove_fib_zk_with(seed, None);
    BatchStarkProver::new(zk_config(seed))
        .with_table_packing(zk_packing(2, 4))
        .verify_all_tables(&p.proof)
        .expect("native verification of a masked ZK proof");
    p
}

/// Test-only mask hook (see `BatchStarkProver::prove_all_tables_with_mask_hook`).
type Hook<'a> = &'a mut dyn FnMut(&mut LookupMaskPlan<Challenge>);

/// [`prove_fib_zk`], optionally through the test-only mask hook, without verifying.
fn prove_fib_zk_with(seed: u64, hook: Option<Hook<'_>>) -> Proved<MyConfigZk> {
    let (circuit, result) = fib_circuit(100);
    let table_packing = zk_packing(2, 4);
    let config = zk_config(seed);
    let (airs_degrees, pc, npc) = get_airs_and_degrees_with_prep::<MyConfigZk, _, 1>(
        &circuit,
        &table_packing,
        &[],
        &[],
        ConstraintProfile::Standard,
    )
    .unwrap();
    let (airs, degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();
    let air_widths = airs.iter().map(BaseAir::<F>::width).collect();
    let ext_degrees: Vec<usize> = degrees.iter().map(|&d| d + config.is_zk()).collect();
    let mut runner = circuit.runner();
    runner.set_public_inputs(&[result]).unwrap();
    let traces = runner.run().unwrap();
    let prover_data = ProverData::from_airs_and_degrees(&config, &airs, &ext_degrees);
    let cpd = CircuitProverData::new(prover_data, pc, npc);
    let prover = BatchStarkProver::new(config).with_table_packing(table_packing);
    let proof = hook
        .map_or_else(
            || prover.prove_all_tables(&traces, &cpd),
            |h| prover.prove_all_tables_with_mask_hook(&traces, &cpd, |p| h(p)),
        )
        .unwrap();
    Proved {
        proof,
        cpd,
        air_widths,
    }
}

/// The same Fibonacci circuit proved without ZK (control: no mask columns).
fn prove_fib_plain() -> Proved<MyConfig> {
    let (circuit, result) = fib_circuit(100);
    let table_packing = TablePacking::new(2, 4);
    let config = make_test_config();
    let (airs_degrees, pc, npc) = get_airs_and_degrees_with_prep::<MyConfig, _, 1>(
        &circuit,
        &table_packing,
        &[],
        &[],
        ConstraintProfile::Standard,
    )
    .unwrap();
    let (airs, degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();
    let air_widths = airs.iter().map(BaseAir::<F>::width).collect();
    let mut runner = circuit.runner();
    runner.set_public_inputs(&[result]).unwrap();
    let traces = runner.run().unwrap();
    let prover_data = ProverData::from_airs_and_degrees(&config, &airs, &degrees);
    let cpd = CircuitProverData::new(prover_data, pc, npc);
    let prover = BatchStarkProver::new(config).with_table_packing(table_packing);
    let proof = prover.prove_all_tables(&traces, &cpd).unwrap();
    prover.verify_all_tables(&proof).expect("native verify");
    Proved {
        proof,
        cpd,
        air_widths,
    }
}

/// Table provers the in-circuit verifier needs to rebuild NPO AIRs of a D=4 verifier circuit.
fn d4_table_provers() -> Vec<Box<dyn TableProver<MyConfigZk>>> {
    let mut tp: Vec<Box<dyn TableProver<MyConfigZk>>> = vec![Box::new(Poseidon2Prover::new(
        POSEIDON2,
        ConstraintProfile::Standard,
    ))];
    tp.extend(recompose_table_provers::<MyConfigZk, 4>(1, true));
    tp
}

fn verifier_circuit_builder() -> CircuitBuilder<Challenge> {
    let mut cb = CircuitBuilder::new();
    cb.enable_poseidon2_perm::<KoalaBearD4Width16, _>(
        generate_poseidon2_trace::<Challenge, KoalaBearD4Width16>,
        default_koalabear_poseidon2_16(),
    );
    cb.enable_recompose::<F>(generate_recompose_trace::<F, Challenge>);
    cb
}

/// An in-circuit verifier built for one proof, with the values that feed it.
struct BuiltVerifier {
    circuit: Circuit<Challenge>,
    public_inputs: Vec<Challenge>,
    private_inputs: Vec<Challenge>,
    mmcs_op_ids: Vec<NonPrimitiveOpId>,
}

fn num_tables<SC: StarkGenericConfig>(cpd: &CircuitProverData<SC>) -> usize {
    cpd.common_data()
        .preprocessed
        .as_ref()
        .map_or(0, |g| g.instances.len())
}

/// Build the in-circuit verifier of a masked ZK proof whose trace field has degree `TRACE_D`.
fn build_verifier_zk<const TRACE_D: usize>(
    inner: &Proved<MyConfigZk>,
    non_primitive_provers: &[Box<dyn TableProver<MyConfigZk>>],
) -> Result<BuiltVerifier, VerificationError> {
    let common = inner.cpd.common_data();
    let mut cb = verifier_circuit_builder();
    let (vi, mmcs_op_ids): (
        BatchStarkVerifierInputsBuilder<MyConfigZk, MerkleCapTargets<F, DIGEST_ELEMS>, InnerFriZk>,
        _,
    ) = verify_p3_batch_proof_circuit::<
        MyConfigZk,
        MerkleCapTargets<F, DIGEST_ELEMS>,
        ValInputProof,
        InnerFriZk,
        LogUpGadget,
        _,
        WIDTH,
        RATE,
        TRACE_D,
    >(
        // The verifier's config: a different RNG seed than the prover's (it is never used to
        // verify), same parameters.
        &zk_config(999),
        &mut cb,
        &inner.proof,
        &fri_verifier_params(),
        common,
        &LogUpGadget::new(),
        POSEIDON2,
        non_primitive_provers,
    )?;
    let circuit = cb.build().expect("verifier circuit builds");
    let pis = vec![vec![]; num_tables(&inner.cpd)];
    let (public_inputs, private_inputs) = vi.pack_values(&pis, &inner.proof.proof, common);
    Ok(BuiltVerifier {
        circuit,
        public_inputs,
        private_inputs,
        mmcs_op_ids,
    })
}

/// Run a built verifier circuit on the proof it was built for; `Ok` iff all its checks pass.
fn run_verifier_zk(
    built: &BuiltVerifier,
    proof: &BatchStarkProof<MyConfigZk>,
) -> Result<Traces<Challenge>, String> {
    let mut runner = built.circuit.runner();
    runner
        .set_public_inputs(&built.public_inputs)
        .map_err(|e| format!("{e:?}"))?;
    runner
        .set_private_inputs(&built.private_inputs)
        .map_err(|e| format!("{e:?}"))?;
    if !built.mmcs_op_ids.is_empty() {
        // HidingFriPcs proofs are (random opened values, inner FRI proof).
        set_fri_mmcs_private_data::<
            F,
            Challenge,
            ChallengeMmcs,
            MyMmcs,
            MyHash,
            MyCompress,
            DIGEST_ELEMS,
        >(
            &mut runner,
            &built.mmcs_op_ids,
            &proof.proof.opening_proof.1,
            POSEIDON2,
        )
        .map_err(|e| format!("{e:?}"))?;
    }
    runner.run().map_err(|e| format!("{e:?}"))
}

fn revealed_sums<SC: StarkGenericConfig>(p: &BatchStarkProof<SC>) -> Vec<SC::Challenge> {
    p.proof
        .global_lookup_data
        .iter()
        .flatten()
        .map(|ld| ld.cumulative_sum)
        .collect()
}

/// Sum of the revealed values per bus name.
fn per_bus_sums(p: &BatchStarkProof<MyConfigZk>) -> BTreeMap<String, Challenge> {
    let mut sums = BTreeMap::new();
    for ld in p.proof.global_lookup_data.iter().flatten() {
        *sums.entry(ld.name.clone()).or_insert(Challenge::ZERO) += ld.cumulative_sum;
    }
    sums
}

fn opened_main_widths<SC: StarkGenericConfig>(p: &BatchStarkProof<SC>) -> Vec<usize> {
    p.proof
        .opened_values
        .instances
        .iter()
        .map(|o| o.base_opened_values.trace_local.len())
        .collect()
}

/// Expected committed main widths: AIR width + `D` per global lookup under ZK.
fn expected_main_widths<SC: StarkGenericConfig>(inner: &Proved<SC>, zk: bool) -> Vec<usize> {
    inner
        .air_widths
        .iter()
        .zip(inner.cpd.common_data().lookups.iter())
        .map(|(w, l)| w + lookup_mask_width(l, D, zk))
        .collect()
}

#[test]
fn zk_masked_proof_verifies_natively_and_in_circuit() {
    let a = prove_fib_zk(1);
    let b = prove_fib_zk(2);

    // Masked layout: D trailing main columns per global lookup, in every instance.
    let widths = opened_main_widths(&a.proof);
    assert_eq!(widths, expected_main_widths(&a, true));
    let globals: Vec<usize> = a
        .cpd
        .common_data()
        .lookups
        .iter()
        .map(|l| num_global_lookups(l))
        .collect();
    assert!(
        globals.iter().filter(|&&g| g > 0).count() >= 2,
        "the fixture must have global lookups in several instances, got {globals:?}"
    );
    assert!(widths.iter().zip(&a.air_widths).any(|(w, aw)| w > aw));
    println!(
        "AIR widths {:?} -> masked main widths {widths:?}",
        a.air_widths
    );

    // Revealed sums: fresh per proof, zero-sum per bus.
    let (sa, sb) = (revealed_sums(&a.proof), revealed_sums(&b.proof));
    assert_eq!(sa.len(), globals.iter().sum::<usize>());
    assert!(
        sa.iter().zip(&sb).all(|(x, y)| x != y),
        "masks must be fresh per proof"
    );
    for (bus, sum) in per_bus_sums(&a.proof) {
        assert_eq!(sum, Challenge::ZERO, "bus {bus} does not balance");
    }

    // In-circuit verification of the masked proof.
    let built = build_verifier_zk::<1>(&a, &[]).expect("verifier circuit for a masked proof");
    run_verifier_zk(&built, &a.proof).expect("in-circuit verification of a masked ZK proof");
}

#[test]
fn nonzk_proof_has_no_mask_columns_and_verifies_in_circuit() {
    let p = prove_fib_plain();
    assert_eq!(opened_main_widths(&p.proof), p.air_widths);
    assert_eq!(expected_main_widths(&p, false), p.air_widths);

    let common = p.cpd.common_data();
    let mut cb = verifier_circuit_builder();
    let (vi, mmcs_op_ids) = verify_p3_batch_proof_circuit::<
        MyConfig,
        MerkleCapTargets<F, DIGEST_ELEMS>,
        ValInputProof,
        InnerFri,
        LogUpGadget,
        _,
        WIDTH,
        RATE,
        1,
    >(
        &make_test_config(),
        &mut cb,
        &p.proof,
        &fri_verifier_params(),
        common,
        &LogUpGadget::new(),
        POSEIDON2,
        &[],
    )
    .expect("verifier circuit for a non-ZK proof");
    let circuit = cb.build().unwrap();
    let pis = vec![vec![]; num_tables(&p.cpd)];
    let (public_inputs, private_inputs) = vi.pack_values(&pis, &p.proof.proof, common);
    let mut runner = circuit.runner();
    runner.set_public_inputs(&public_inputs).unwrap();
    runner.set_private_inputs(&private_inputs).unwrap();
    if !mmcs_op_ids.is_empty() {
        set_fri_mmcs_private_data::<
            F,
            Challenge,
            ChallengeMmcs,
            MyMmcs,
            MyHash,
            MyCompress,
            DIGEST_ELEMS,
        >(
            &mut runner,
            &mmcs_op_ids,
            &p.proof.proof.opening_proof,
            POSEIDON2,
        )
        .unwrap();
    }
    runner
        .run()
        .expect("in-circuit verification of a non-ZK proof");
}

/// Moving `delta` between two revealed sums of the same bus after proving keeps every per-bus
/// total (so the per-bus check passes). The revealed sums are absorbed into the transcript
/// before the constraint challenge, so this edit also desynchronizes the challenges from the
/// proof: the rejection here does not by itself show that the masked last-row binding is
/// enforced. `zk_masked_in_circuit_rejects_transcript_consistent_same_bus_shift` isolates the
/// binding.
#[test]
fn zk_masked_in_circuit_rejects_same_bus_tamper() {
    let mut tampered = prove_fib_zk(3);
    let honest_bus_sums = per_bus_sums(&tampered.proof);
    let delta = Challenge::from_u64(5);

    let entries: Vec<(usize, usize, String)> = tampered
        .proof
        .proof
        .global_lookup_data
        .iter()
        .enumerate()
        .flat_map(|(i, v)| {
            v.iter()
                .enumerate()
                .map(move |(g, ld)| (i, g, ld.name.clone()))
        })
        .collect();
    let (i0, g0, bus) = entries[0].clone();
    let &(i1, g1, _) = entries
        .iter()
        .find(|(i, _, n)| *n == bus && *i != i0)
        .expect("a second instance on the same bus");

    tampered.proof.proof.global_lookup_data[i0][g0].cumulative_sum += delta;
    tampered.proof.proof.global_lookup_data[i1][g1].cumulative_sum -= delta;
    assert_eq!(per_bus_sums(&tampered.proof), honest_bus_sums);

    // Native verifier rejects.
    let prover = BatchStarkProver::new(zk_config(4)).with_table_packing(zk_packing(2, 4));
    assert!(prover.verify_all_tables(&tampered.proof).is_err());

    // In-circuit verifier: the circuit builds (the sums are witness values) but does not run.
    let built = build_verifier_zk::<1>(&tampered, &[]).expect("shape is unchanged");
    let res = run_verifier_zk(&built, &tampered.proof);
    println!("tampered in-circuit result: {res:?}");
    let err = res.expect_err("in-circuit verifier must reject a same-bus tamper");
    assert!(err.contains("WitnessConflict"), "unexpected error: {err}");
}

/// The same `+delta` / `-delta` shift on two WitnessChecks members, made by the prover through
/// the test-only mask hook: the honest masks are committed, the shifted sums are revealed and
/// absorbed into the transcript, so the challenges match the proof and the bus balances. Only
/// the masked last-row binding `(c' - rho - s[n-1]) * den - num = 0` is violated: natively an
/// out-of-domain mismatch, in circuit a conflict in the quotient identity.
#[test]
fn zk_masked_in_circuit_rejects_transcript_consistent_same_bus_shift() {
    let seed = 7;
    let delta = Challenge::from_u64(5);
    let honest = prove_fib_zk(seed);
    let entries: Vec<(usize, usize)> = honest
        .proof
        .proof
        .global_lookup_data
        .iter()
        .enumerate()
        .flat_map(|(i, v)| (0..v.len()).map(move |g| (i, g)))
        .collect();
    let a = entries[0];
    let b = *entries
        .iter()
        .find(|(i, _)| *i != a.0)
        .expect("a second table on WitnessChecks");
    let mut hook = |plan: &mut LookupMaskPlan<Challenge>| {
        plan.reveal_offsets[a.0][a.1] += delta;
        plan.reveal_offsets[b.0][b.1] -= delta;
    };
    let forged = prove_fib_zk_with(seed, Some(&mut hook));

    // Same seed, same masks and challenges: exactly the two revealed sums moved.
    let (h, f) = (revealed_sums(&honest.proof), revealed_sums(&forged.proof));
    let flat = |(i, g): (usize, usize)| entries.iter().position(|&e| e == (i, g)).unwrap();
    for (k, (x, y)) in h.iter().zip(&f).enumerate() {
        let expected = if k == flat(a) {
            *x + delta
        } else if k == flat(b) {
            *x - delta
        } else {
            *x
        };
        assert_eq!(*y, expected, "revealed sum {k}");
    }
    assert_eq!(per_bus_sums(&forged.proof), per_bus_sums(&honest.proof));

    let prover = BatchStarkProver::new(zk_config(8)).with_table_packing(zk_packing(2, 4));
    let err = prover
        .verify_all_tables(&forged.proof)
        .expect_err("native verifier must reject");
    let msg = format!("{err:?}");
    assert!(msg.contains("OodEvaluationMismatch"), "{msg}");

    let built = build_verifier_zk::<1>(&forged, &[]).expect("shape is unchanged");
    let err = run_verifier_zk(&built, &forged.proof)
        .expect_err("in-circuit verifier must reject a transcript-consistent same-bus shift");
    println!("transcript-consistent forgery, in circuit: {err}");
    assert!(err.contains("WitnessConflict"), "unexpected error: {err}");
    // Not the per-bus check (which would conflict with the bus total, here zero anyway).
    assert!(!err.contains("WitnessId(0)"), "unexpected error: {err}");
}

/// The masked mode is fixed by the verifier's own width; stripping the mask columns from the
/// openings is a shape error, natively and in circuit.
#[test]
fn zk_masked_in_circuit_rejects_stripped_mask_columns() {
    let mut p = prove_fib_zk(5);
    let lookups = p.cpd.common_data().lookups.clone();
    let (i, mask_w) = lookups
        .iter()
        .enumerate()
        .map(|(i, l)| (i, lookup_mask_width(l, D, true)))
        .find(|&(_, w)| w > 0)
        .unwrap();
    let ov = &mut p.proof.proof.opened_values.instances[i].base_opened_values;
    let plain_w = ov.trace_local.len() - mask_w;
    assert_eq!(plain_w, p.air_widths[i]);
    ov.trace_local.truncate(plain_w);
    ov.trace_next
        .as_mut()
        .expect("trace_next opened")
        .truncate(plain_w);

    let prover = BatchStarkProver::new(zk_config(6)).with_table_packing(zk_packing(2, 4));
    assert!(prover.verify_all_tables(&p.proof).is_err());

    match build_verifier_zk::<1>(&p, &[]) {
        Err(VerificationError::InvalidProofShape(msg)) => {
            assert!(msg.contains("incorrect trace width"), "{msg}");
        }
        Err(e) => panic!("unexpected error {e:?}"),
        Ok(_) => panic!("stripped mask columns must be rejected"),
    }
}

fn first_op_diff(a: &Circuit<Challenge>, b: &Circuit<Challenge>) -> Option<usize> {
    if a.ops.len() != b.ops.len() {
        return Some(a.ops.len().min(b.ops.len()));
    }
    a.ops.iter().zip(b.ops.iter()).position(|(x, y)| x != y)
}

/// The verifier circuit (its op list, i.e. its verification key) is a function of the proof
/// shape only: fresh masks must not change it.
#[test]
fn zk_masked_verifier_circuit_is_mask_independent() {
    let a = prove_fib_zk(21);
    let b = prove_fib_zk(22);
    assert_ne!(revealed_sums(&a.proof), revealed_sums(&b.proof));

    let va = build_verifier_zk::<1>(&a, &[]).unwrap();
    let vb = build_verifier_zk::<1>(&b, &[]).unwrap();
    assert_eq!(
        first_op_diff(&va.circuit, &vb.circuit),
        None,
        "verifier circuit differs between two proofs with different masks"
    );
    // And rebuilding from the same proof is deterministic.
    for _ in 0..4 {
        let again = build_verifier_zk::<1>(&a, &[]).unwrap();
        assert_eq!(first_op_diff(&va.circuit, &again.circuit), None);
    }
    println!("verifier circuit ops: {}", va.circuit.ops.len());
    run_verifier_zk(&va, &a.proof).unwrap();
    run_verifier_zk(&vb, &b.proof).unwrap();
}

/// Prove a D=4 circuit (an in-circuit verifier: Poseidon2, recompose and packed-Horner ALU
/// tables, all with global lookups) with ZK, and verify it natively.
fn prove_verifier_circuit_zk(
    circuit: &Circuit<Challenge>,
    traces: &Traces<Challenge>,
    seed: u64,
) -> Proved<MyConfigZk> {
    let packing = zk_packing(1, 8);
    let npo_prep: Vec<Box<dyn NpoPreprocessor<F>>> = vec![
        Box::new(Poseidon2Preprocessor),
        Box::new(RecomposePreprocessor::new(true)),
    ];
    let mut air_builders = poseidon2_air_builders::<_, 4>();
    air_builders.extend(recompose_air_builders(1, true));
    let (airs_degrees, pc, npc) = get_airs_and_degrees_with_prep::<MyConfigZk, _, 4>(
        circuit,
        &packing,
        &npo_prep,
        &air_builders,
        ConstraintProfile::Standard,
    )
    .unwrap();
    let (airs, degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();
    let air_widths = airs.iter().map(BaseAir::<F>::width).collect();
    let config = zk_config(seed);
    let ext_degrees: Vec<usize> = degrees.iter().map(|&d| d + config.is_zk()).collect();
    let prover_data = ProverData::from_airs_and_degrees(&config, &airs, &ext_degrees);
    let cpd = CircuitProverData::new(prover_data, pc, npc);
    let mut prover = BatchStarkProver::new(config).with_table_packing(packing);
    prover.register_poseidon2_table::<4>(POSEIDON2);
    prover.register_recompose_table::<4>(true);
    let proof = prover.prove_all_tables(traces, &cpd).unwrap();
    prover
        .verify_all_tables(&proof)
        .expect("native verification of a masked ZK proof with NPO tables");
    Proved {
        proof,
        cpd,
        air_widths,
    }
}

/// ZK leaf -> in-circuit verify -> ZK proof of that verifier (masked Poseidon2, recompose and
/// ALU tables) -> in-circuit verify again -> ZK proof -> native verification.
#[test]
fn zk_masked_recursion_chain_with_npo_tables() {
    // Level 0: masked ZK leaf.
    let leaf = prove_fib_zk(31);
    let v1 = build_verifier_zk::<1>(&leaf, &[]).unwrap();
    let t1 = run_verifier_zk(&v1, &leaf.proof).expect("level-1 verifier runs");

    // Level 1: prove the leaf verifier with ZK.
    let l1 = prove_verifier_circuit_zk(&v1.circuit, &t1, 32);
    assert_eq!(
        opened_main_widths(&l1.proof),
        expected_main_widths(&l1, true)
    );
    for (bus, sum) in per_bus_sums(&l1.proof) {
        assert_eq!(sum, Challenge::ZERO, "bus {bus} does not balance");
    }
    assert!(
        !l1.proof.non_primitives.is_empty(),
        "level 1 must carry NPO tables"
    );

    // Level 2: verify the masked level-1 proof in circuit, then prove that with ZK.
    let v2 = build_verifier_zk::<4>(&l1, &d4_table_provers()).unwrap();
    let t2 = run_verifier_zk(&v2, &l1.proof).expect("in-circuit verification of level 1");
    let l2 = prove_verifier_circuit_zk(&v2.circuit, &t2, 33);
    println!(
        "level-1 widths {:?} -> {:?}; level-2 widths {:?} -> {:?}",
        l1.air_widths,
        opened_main_widths(&l1.proof),
        l2.air_widths,
        opened_main_widths(&l2.proof)
    );
}
