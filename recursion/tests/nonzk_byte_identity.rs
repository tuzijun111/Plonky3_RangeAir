//! Without ZK the LogUp-mask changes must leave everything byte-identical: the proofs the
//! circuit prover produces and the in-circuit verifier circuits built for them.
//!
//! The golden values were produced by this fork at fbcc78f against Plonky3 56952503 (before the
//! masks). They depend on the transcript, the AIRs and the proof serialization, so any change of
//! the non-ZK path shows up here. The FRI proof-of-work bits are 0 (the grinding witness is then
//! fixed), so the proofs are deterministic with and without the `parallel` feature.

mod common;

use p3_batch_stark::ProverData;
use p3_circuit::ops::{generate_poseidon2_trace, generate_recompose_trace};
use p3_circuit::{Circuit, CircuitBuilder, Traces};
use p3_circuit_prover::batch_stark_prover::{
    BatchStarkProof, poseidon2_air_builders, recompose_air_builders,
};
use p3_circuit_prover::common::{NpoPreprocessor, get_airs_and_degrees_with_prep};
use p3_circuit_prover::{
    BatchStarkProver, CircuitProverData, ConstraintProfile, Poseidon2Preprocessor,
    RecomposePreprocessor, TablePacking,
};
use p3_lookup::logup::LogUpGadget;
use p3_poseidon2_circuit_air::KoalaBearD4Width16;
use p3_recursion::Poseidon2Config;
use p3_recursion::pcs::fri::{FriVerifierParams, InputProofTargets, MerkleCapTargets, RecValMmcs};
use p3_recursion::pcs::set_fri_mmcs_private_data;
use p3_recursion::verifier::verify_p3_batch_proof_circuit;
use p3_test_utils::koala_bear_params::*;
use p3_test_utils::test_fri_scalars;

use crate::common::InnerFriGeneric;

type InnerFri = InnerFriGeneric<MyConfig, MyHash, MyCompress, DIGEST_ELEMS>;

const POSEIDON2: Poseidon2Config = Poseidon2Config::KOALA_BEAR_D4_W16;

/// FNV-1a, 64 bit.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// The test config with the proof-of-work bits set to 0: grinding then returns a fixed witness
/// instead of the first one some thread finds, so proofs are deterministic in parallel builds.
fn config() -> MyConfig {
    let perm = default_koalabear_poseidon2_16();
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = MyMmcs::new(hash, compress, 0);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let fri_params = FriParameters {
        commit_proof_of_work_bits: 0,
        query_proof_of_work_bits: 0,
        ..FriParameters::new_testing(challenge_mmcs, 0)
    };
    let pcs = MyPcs::new(Dft::default(), val_mmcs, fri_params);
    MyConfig::new(pcs, Challenger::new(perm))
}

fn proof_digest(p: &BatchStarkProof<MyConfig>) -> (usize, u64) {
    let bytes = postcard::to_allocvec(p).unwrap();
    (bytes.len(), fnv1a(&bytes))
}

fn ops_digest(c: &Circuit<Challenge>) -> (usize, u64) {
    (c.ops.len(), fnv1a(format!("{:?}", c.ops).as_bytes()))
}

/// Non-ZK Fibonacci proof (Const/Public/ALU tables).
fn fib_proof() -> (BatchStarkProof<MyConfig>, CircuitProverData<MyConfig>) {
    let n: usize = 100;
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
    let circuit = builder.build().unwrap();
    let table_packing = TablePacking::new(2, 4);
    let config = config();
    let (airs_degrees, pc, npc) = get_airs_and_degrees_with_prep::<MyConfig, _, 1>(
        &circuit,
        &table_packing,
        &[],
        &[],
        ConstraintProfile::Standard,
    )
    .unwrap();
    let (airs, degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();
    let mut runner = circuit.runner();
    runner.set_public_inputs(&[x.1]).unwrap();
    let traces = runner.run().unwrap();
    let prover_data = ProverData::from_airs_and_degrees(&config, &airs, &degrees);
    let cpd = CircuitProverData::new(prover_data, pc, npc);
    let prover = BatchStarkProver::new(config).with_table_packing(table_packing);
    let proof = prover.prove_all_tables(&traces, &cpd).unwrap();
    prover.verify_all_tables(&proof).unwrap();
    (proof, cpd)
}

/// The in-circuit verifier of the Fibonacci proof, and its traces.
fn verifier_circuit(
    proof: &BatchStarkProof<MyConfig>,
    cpd: &CircuitProverData<MyConfig>,
) -> (Circuit<Challenge>, Traces<Challenge>) {
    let common = cpd.common_data();
    let s = test_fri_scalars();
    let fvp = FriVerifierParams::with_mmcs(
        s.log_blowup,
        s.log_final_poly_len,
        0,
        0,
        s.num_queries,
        POSEIDON2,
    );
    let mut cb = CircuitBuilder::new();
    cb.enable_poseidon2_perm::<KoalaBearD4Width16, _>(
        generate_poseidon2_trace::<Challenge, KoalaBearD4Width16>,
        default_koalabear_poseidon2_16(),
    );
    cb.enable_recompose::<F>(generate_recompose_trace::<F, Challenge>);
    let (vi, ids) = verify_p3_batch_proof_circuit::<
        MyConfig,
        MerkleCapTargets<F, DIGEST_ELEMS>,
        InputProofTargets<F, Challenge, RecValMmcs<F, DIGEST_ELEMS, MyHash, MyCompress>>,
        InnerFri,
        LogUpGadget,
        _,
        WIDTH,
        RATE,
        1,
    >(
        &config(),
        &mut cb,
        proof,
        &fvp,
        common,
        &LogUpGadget::new(),
        POSEIDON2,
        &[],
    )
    .unwrap();
    let vc = cb.build().unwrap();
    let num_tables = common
        .preprocessed
        .as_ref()
        .map_or(0, |g| g.instances.len());
    let pis: Vec<Vec<F>> = vec![vec![]; num_tables];
    let (pubi, privi) = vi.pack_values(&pis, &proof.proof, common);
    let traces = {
        let mut r = vc.runner();
        r.set_public_inputs(&pubi).unwrap();
        r.set_private_inputs(&privi).unwrap();
        set_fri_mmcs_private_data::<
            F,
            Challenge,
            ChallengeMmcs,
            MyMmcs,
            MyHash,
            MyCompress,
            DIGEST_ELEMS,
        >(&mut r, &ids, &proof.proof.opening_proof, POSEIDON2)
        .unwrap();
        r.run().unwrap()
    };
    (vc, traces)
}

/// Non-ZK proof of the verifier circuit (Const/Public/ALU/Poseidon2/recompose tables).
fn verifier_circuit_proof(
    vc: &Circuit<Challenge>,
    traces: &Traces<Challenge>,
) -> BatchStarkProof<MyConfig> {
    let packing = TablePacking::new(1, 8);
    let config = config();
    let npo_prep: Vec<Box<dyn NpoPreprocessor<F>>> = vec![
        Box::new(Poseidon2Preprocessor),
        Box::new(RecomposePreprocessor::new(true)),
    ];
    let mut air_builders = poseidon2_air_builders::<_, 4>();
    air_builders.extend(recompose_air_builders(1, true));
    let (airs_degrees, pc, npc) = get_airs_and_degrees_with_prep::<MyConfig, _, 4>(
        vc,
        &packing,
        &npo_prep,
        &air_builders,
        ConstraintProfile::Standard,
    )
    .unwrap();
    let (airs, degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();
    let prover_data = ProverData::from_airs_and_degrees(&config, &airs, &degrees);
    let cpd = CircuitProverData::new(prover_data, pc, npc);
    let mut prover = BatchStarkProver::new(config).with_table_packing(packing);
    prover.register_poseidon2_table::<4>(POSEIDON2);
    prover.register_recompose_table::<4>(true);
    let proof = prover.prove_all_tables(traces, &cpd).unwrap();
    prover.verify_all_tables(&proof).unwrap();
    proof
}

#[test]
fn nonzk_proofs_and_verifier_circuits_are_unchanged() {
    let (fib, cpd) = fib_proof();
    let (vc, traces) = verifier_circuit(&fib, &cpd);
    let level1 = verifier_circuit_proof(&vc, &traces);

    let got = [proof_digest(&fib), ops_digest(&vc), proof_digest(&level1)];
    println!("non-ZK digests (len, fnv1a64): {got:x?}");

    // Determinism of the fixture itself.
    let (fib2, cpd2) = fib_proof();
    let (vc2, traces2) = verifier_circuit(&fib2, &cpd2);
    assert_eq!(got[0], proof_digest(&fib2));
    assert_eq!(got[1], ops_digest(&vc2));
    assert_eq!(
        got[2],
        proof_digest(&verifier_circuit_proof(&vc2, &traces2))
    );

    assert_eq!(got, GOLDEN, "the non-ZK path changed");
}

/// (byte length or op count, FNV-1a 64) of: the Fibonacci proof, its verifier circuit's ops,
/// and the proof of that verifier circuit; from fbcc78f on Plonky3 56952503.
const GOLDEN: [(usize, u64); 3] = [
    (7532, 0x6b6d_10b7_55e1_1fcd),
    (3351, 0x1885_0091_6726_81dc),
    (43257, 0x013d_1269_2517_3eba),
];
