mod common;

use p3_batch_stark::ProverData;
use p3_circuit::CircuitBuilder;
use p3_circuit::ops::{generate_poseidon2_trace, generate_recompose_trace};
use p3_circuit_prover::batch_stark_prover::{poseidon2_air_builders, recompose_air_builders};
use p3_circuit_prover::common::{NpoPreprocessor, get_airs_and_degrees_with_prep};
use p3_circuit_prover::{
    BatchStarkProver, CircuitProverData, ConstraintProfile, Poseidon2Preprocessor, Poseidon2Prover,
    RecomposePreprocessor, TablePacking, TableProver, recompose_table_provers,
};
use p3_lookup::logup::LogUpGadget;
use p3_poseidon2_circuit_air::KoalaBearD4Width16;
use p3_recursion::Poseidon2Config;
use p3_recursion::pcs::fri::{FriVerifierParams, InputProofTargets, MerkleCapTargets, RecValMmcs};
use p3_recursion::pcs::set_fri_mmcs_private_data;
use p3_recursion::verifier::verify_p3_batch_proof_circuit;
use p3_test_utils::koala_bear_params::*;
use tracing_forest::ForestLayer;
use tracing_forest::util::LevelFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry};

use crate::common::InnerFriGeneric;

type InnerFri = InnerFriGeneric<MyConfig, MyHash, MyCompress, DIGEST_ELEMS>;

fn init_logger() {
    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .init();
}

#[test]
fn test_fibonacci_batch_verifier() {
    init_logger();

    let n: usize = 100;

    let mut builder = CircuitBuilder::new();

    // Public input: expected F(n)
    let expected_result = builder.alloc_public_input("expected_result");

    // Compute F(n) iteratively
    let mut a = builder.alloc_const(F::ZERO, "F(0)");
    let mut b = builder.alloc_const(F::ONE, "F(1)");

    for _i in 2..=n {
        let next = builder.add(a, b);
        a = b;
        b = next;
    }

    // Assert computed F(n) equals expected result
    builder.connect(b, expected_result);

    builder.dump_allocation_log();

    let table_packing = TablePacking::new(2, 4);

    // Use the default permutation for proving to match circuit's Fiat-Shamir challenger
    let config_proving = make_test_config();

    let circuit = builder.build().unwrap();
    let (airs_degrees, primitive_columns, non_primitive_columns) =
        get_airs_and_degrees_with_prep::<MyConfig, _, 1>(
            &circuit,
            &table_packing,
            &[],
            &[],
            ConstraintProfile::Standard,
        )
        .unwrap();
    let (airs, degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();
    let mut runner = circuit.runner();

    // Set public input
    let expected_fib = compute_fibonacci_classical(n);
    runner.set_public_inputs(&[expected_fib]).unwrap();

    let traces = runner.run().unwrap();

    // Create prover data for proving and verifying.
    let prover_data = ProverData::from_airs_and_degrees(&config_proving, &airs, &degrees);
    let circuit_prover_data =
        CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);

    let prover = BatchStarkProver::new(config_proving).with_table_packing(table_packing);

    let lookup_gadget = LogUpGadget::new();
    let batch_stark_proof = prover
        .prove_all_tables(&traces, &circuit_prover_data)
        .unwrap();

    let common = circuit_prover_data.common_data();
    prover.verify_all_tables(&batch_stark_proof).unwrap();

    // Now verify the batch STARK proof recursively
    // Use same permutation as proving to ensure Fiat-Shamir transcript compatibility
    let scalars = test_fri_scalars();
    let fri_verifier_params = FriVerifierParams::with_mmcs(
        scalars.log_blowup,
        scalars.log_final_poly_len,
        scalars.commit_pow_bits,
        scalars.query_pow_bits,
        scalars.num_queries,
        Poseidon2Config::KOALA_BEAR_D4_W16,
    );
    let config = make_test_config();

    // Extract proof components
    let batch_proof = &batch_stark_proof.proof;

    const TRACE_D: usize = 1; // Proof traces are in base field

    // Public values (empty for all 5 circuit tables: Witness, Const, Public, Alu, Poseidon2)
    let num_tables = common
        .preprocessed
        .as_ref()
        .map(|g| g.instances.len())
        .unwrap_or(0);
    let pis: Vec<Vec<F>> = vec![vec![]; num_tables];

    // Build the recursive verification circuit
    let mut circuit_builder = CircuitBuilder::new();
    let poseidon2_perm = default_koalabear_poseidon2_16();
    circuit_builder.enable_poseidon2_perm::<KoalaBearD4Width16, _>(
        generate_poseidon2_trace::<Challenge, KoalaBearD4Width16>,
        poseidon2_perm,
    );
    circuit_builder.enable_recompose::<F>(generate_recompose_trace::<F, Challenge>);

    // Attach verifier without manually building circuit_airs
    let (verifier_inputs, mmcs_op_ids) = verify_p3_batch_proof_circuit::<
        MyConfig,
        MerkleCapTargets<F, DIGEST_ELEMS>,
        InputProofTargets<F, Challenge, RecValMmcs<F, DIGEST_ELEMS, MyHash, MyCompress>>,
        InnerFri,
        LogUpGadget,
        _,
        WIDTH,
        RATE,
        TRACE_D,
    >(
        &config,
        &mut circuit_builder,
        &batch_stark_proof,
        &fri_verifier_params,
        common,
        &lookup_gadget,
        Poseidon2Config::KOALA_BEAR_D4_W16,
        &{
            let mut tp: Vec<Box<dyn TableProver<MyConfig>>> = vec![Box::new(Poseidon2Prover::new(
                Poseidon2Config::KOALA_BEAR_D4_W16,
                ConstraintProfile::Standard,
            ))];
            tp.extend(recompose_table_provers::<_, 4>(1, false));
            tp
        },
    )
    .unwrap();

    // Build the circuit
    let verification_circuit = circuit_builder.build().unwrap();
    let expected_public_input_len = verification_circuit.public_flat_len;

    // Pack values using the builder
    let (public_inputs, private_inputs) = verifier_inputs.pack_values(&pis, batch_proof, common);

    assert_eq!(public_inputs.len(), expected_public_input_len);
    assert!(!public_inputs.is_empty());

    let verification_table_packing = TablePacking::new(1, 8);
    let poseidon2_config = Poseidon2Config::KOALA_BEAR_D4_W16;
    let npo_prep: Vec<Box<dyn NpoPreprocessor<F>>> = vec![
        Box::new(Poseidon2Preprocessor),
        Box::new(RecomposePreprocessor::default()),
    ];
    let mut air_builders = poseidon2_air_builders::<_, 4>();
    air_builders.extend(recompose_air_builders(1, false));
    let (
        verification_airs_degrees,
        verification_primitive_columns,
        verification_non_primitive_columns,
    ) = get_airs_and_degrees_with_prep::<MyConfig, _, 4>(
        &verification_circuit,
        &verification_table_packing,
        &npo_prep,
        &air_builders,
        ConstraintProfile::Standard,
    )
    .unwrap();
    let (verification_airs, verification_degrees): (Vec<_>, Vec<usize>) =
        verification_airs_degrees.into_iter().unzip();

    // Now run the circuit to generate traces
    let mut runner = verification_circuit.runner();
    runner.set_public_inputs(&public_inputs).unwrap();
    runner.set_private_inputs(&private_inputs).unwrap();

    // Set MMCS private data for the verification circuit
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
        &batch_stark_proof.proof.opening_proof,
        Poseidon2Config::KOALA_BEAR_D4_W16,
    )
    .unwrap();

    // Run the circuit to generate traces
    let verification_traces = runner.run().unwrap();

    // Create a new config and prover for the verification circuit
    let config3 = make_test_config();

    let verification_prover_data =
        ProverData::from_airs_and_degrees(&config3, &verification_airs, &verification_degrees);
    let verification_circuit_prover_data = CircuitProverData::new(
        verification_prover_data,
        verification_primitive_columns,
        verification_non_primitive_columns,
    );

    let mut verification_prover =
        BatchStarkProver::new(config3).with_table_packing(verification_table_packing);
    verification_prover.register_poseidon2_table::<4>(poseidon2_config);
    verification_prover.register_recompose_table::<4>(false);

    // Prove the verification circuit
    let verification_proof = verification_prover
        .prove_all_tables(&verification_traces, &verification_circuit_prover_data)
        .expect("Failed to prove verification circuit");

    // Verify the proof of the verification circuit
    verification_prover
        .verify_all_tables(&verification_proof)
        .expect("Failed to verify proof of verification circuit");
}

/// Regression for upstream e8a29ba: the in-circuit batch-STARK verifier must rebuild the
/// lookup contexts from the AIRs it reconstructs, not believe the `common.lookups` handed to
/// it alongside the proof. A right-length/wrong-content lookup set (here: the ALU table's
/// lookup set emptied) must produce exactly the same verifier circuit as the honest one.
#[test]
fn recursive_verifier_ignores_supplied_lookups() {
    use p3_batch_stark::CommonData;
    use p3_batch_stark::common::GlobalPreprocessed;

    let n: usize = 20;
    let mut builder = CircuitBuilder::new();
    let expected_result = builder.alloc_public_input("expected_result");
    let mut a = builder.alloc_const(F::ZERO, "F(0)");
    let mut b = builder.alloc_const(F::ONE, "F(1)");
    for _i in 2..=n {
        let next = builder.add(a, b);
        a = b;
        b = next;
    }
    builder.connect(b, expected_result);
    let circuit = builder.build().unwrap();

    let table_packing = TablePacking::new(2, 4);
    let config_proving = make_test_config();
    let (airs_degrees, primitive_columns, non_primitive_columns) =
        get_airs_and_degrees_with_prep::<MyConfig, _, 1>(
            &circuit,
            &table_packing,
            &[],
            &[],
            ConstraintProfile::Standard,
        )
        .unwrap();
    let (airs, degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();
    let mut runner = circuit.runner();
    runner
        .set_public_inputs(&[compute_fibonacci_classical(n)])
        .unwrap();
    let traces = runner.run().unwrap();
    let prover_data = ProverData::from_airs_and_degrees(&config_proving, &airs, &degrees);
    let circuit_prover_data =
        CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);
    let prover = BatchStarkProver::new(config_proving).with_table_packing(table_packing);
    let batch_stark_proof = prover
        .prove_all_tables(&traces, &circuit_prover_data)
        .unwrap();
    prover.verify_all_tables(&batch_stark_proof).unwrap();
    let honest_common = circuit_prover_data.common_data();

    // Same preprocessed binding and instance count, but the ALU table's lookup set emptied:
    // the right-length/wrong-content shape that collapses one AIR's lookup argument.
    let mut lookups = honest_common.lookups.clone();
    assert!(lookups.len() >= 3 && !lookups[2].is_empty());
    // Collapse the ALU table's lookup argument (index 2: Const, Public, Alu, ...).
    lookups[2] = Default::default();
    let tampered_common = CommonData::<MyConfig>::new(
        honest_common
            .preprocessed
            .as_ref()
            .map(|g| GlobalPreprocessed {
                commitment: g.commitment.clone(),
                instances: g.instances.clone(),
                matrix_to_instance: g.matrix_to_instance.clone(),
            }),
        lookups,
    );

    let scalars = test_fri_scalars();
    let fri_verifier_params = FriVerifierParams::with_mmcs(
        scalars.log_blowup,
        scalars.log_final_poly_len,
        scalars.commit_pow_bits,
        scalars.query_pow_bits,
        scalars.num_queries,
        Poseidon2Config::KOALA_BEAR_D4_W16,
    );
    let config = make_test_config();
    let lookup_gadget = LogUpGadget::new();

    let build = |common: &CommonData<MyConfig>| {
        let mut circuit_builder = CircuitBuilder::new();
        circuit_builder.enable_poseidon2_perm::<KoalaBearD4Width16, _>(
            generate_poseidon2_trace::<Challenge, KoalaBearD4Width16>,
            default_koalabear_poseidon2_16(),
        );
        circuit_builder.enable_recompose::<F>(generate_recompose_trace::<F, Challenge>);
        let mut tp: Vec<Box<dyn TableProver<MyConfig>>> = vec![Box::new(Poseidon2Prover::new(
            Poseidon2Config::KOALA_BEAR_D4_W16,
            ConstraintProfile::Standard,
        ))];
        tp.extend(recompose_table_provers::<_, 4>(1, false));
        verify_p3_batch_proof_circuit::<
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
            &config,
            &mut circuit_builder,
            &batch_stark_proof,
            &fri_verifier_params,
            common,
            &lookup_gadget,
            Poseidon2Config::KOALA_BEAR_D4_W16,
            &tp,
        )
        .map_err(|e| format!("{e:?}"))?;
        circuit_builder
            .build()
            .map_err(|e| format!("{e:?}"))?
            .generate_preprocessed_columns::<4>()
            .map_err(|e| format!("{e:?}"))
    };

    let honest = build(honest_common).expect("honest verifier circuit builds");
    let tampered = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| build(&tampered_common)))
        .unwrap_or_else(|_| Err("verifier circuit construction panicked".into()));
    match &tampered {
        Ok(t) => assert!(
            *t == honest,
            "a collapsed lookup set changed the verifier circuit: the verifier believed it"
        ),
        Err(e) => panic!("a collapsed lookup set was not ignored by the verifier: {e}"),
    }
}

/// Build, run, prove and verify the recursive verifier circuit for `proof`.
fn verify_fib_batch_proof_recursively(
    proof: &p3_circuit_prover::batch_stark_prover::BatchStarkProof<MyConfig>,
    common: &p3_batch_stark::CommonData<MyConfig>,
    fri_verifier_params: &FriVerifierParams,
) -> Result<(), String> {
    let config = make_test_config();
    let lookup_gadget = LogUpGadget::new();
    let num_tables = common
        .preprocessed
        .as_ref()
        .map(|g| g.instances.len())
        .unwrap_or(0);
    let pis: Vec<Vec<F>> = vec![vec![]; num_tables];

    let mut circuit_builder = CircuitBuilder::new();
    circuit_builder.enable_poseidon2_perm::<KoalaBearD4Width16, _>(
        generate_poseidon2_trace::<Challenge, KoalaBearD4Width16>,
        default_koalabear_poseidon2_16(),
    );
    circuit_builder.enable_recompose::<F>(generate_recompose_trace::<F, Challenge>);
    let mut tp: Vec<Box<dyn TableProver<MyConfig>>> = vec![Box::new(Poseidon2Prover::new(
        Poseidon2Config::KOALA_BEAR_D4_W16,
        ConstraintProfile::Standard,
    ))];
    tp.extend(recompose_table_provers::<_, 4>(1, false));
    let (verifier_inputs, mmcs_op_ids) = verify_p3_batch_proof_circuit::<
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
        &config,
        &mut circuit_builder,
        proof,
        fri_verifier_params,
        common,
        &lookup_gadget,
        Poseidon2Config::KOALA_BEAR_D4_W16,
        &tp,
    )
    .map_err(|e| format!("verifier circuit construction: {e:?}"))?;
    let verification_circuit = circuit_builder
        .build()
        .map_err(|e| format!("build: {e:?}"))?;
    let (public_inputs, private_inputs) = verifier_inputs.pack_values(&pis, &proof.proof, common);

    let verification_table_packing = TablePacking::new(1, 8);
    let npo_prep: Vec<Box<dyn NpoPreprocessor<F>>> = vec![
        Box::new(Poseidon2Preprocessor),
        Box::new(RecomposePreprocessor::default()),
    ];
    let mut air_builders = poseidon2_air_builders::<_, 4>();
    air_builders.extend(recompose_air_builders(1, false));
    let (vad, vpc, vnpc) = get_airs_and_degrees_with_prep::<MyConfig, _, 4>(
        &verification_circuit,
        &verification_table_packing,
        &npo_prep,
        &air_builders,
        ConstraintProfile::Standard,
    )
    .map_err(|e| format!("airs: {e:?}"))?;
    let (vairs, vdegs): (Vec<_>, Vec<usize>) = vad.into_iter().unzip();

    let mut runner = verification_circuit.runner();
    runner
        .set_public_inputs(&public_inputs)
        .map_err(|e| format!("{e:?}"))?;
    runner
        .set_private_inputs(&private_inputs)
        .map_err(|e| format!("{e:?}"))?;
    set_fri_mmcs_private_data::<F, Challenge, ChallengeMmcs, MyMmcs, MyHash, MyCompress, DIGEST_ELEMS>(
        &mut runner,
        &mmcs_op_ids,
        &proof.proof.opening_proof,
        Poseidon2Config::KOALA_BEAR_D4_W16,
    )
    .map_err(|e| e.to_string())?;
    let traces = runner.run().map_err(|e| format!("run: {e:?}"))?;

    let config3 = make_test_config();
    let vpd = ProverData::from_airs_and_degrees(&config3, &vairs, &vdegs);
    let vcpd = CircuitProverData::new(vpd, vpc, vnpc);
    let mut vprover = BatchStarkProver::new(config3).with_table_packing(verification_table_packing);
    vprover.register_poseidon2_table::<4>(Poseidon2Config::KOALA_BEAR_D4_W16);
    vprover.register_recompose_table::<4>(false);
    let vproof = vprover
        .prove_all_tables(&traces, &vcpd)
        .map_err(|e| format!("prove: {e:?}"))?;
    vprover
        .verify_all_tables(&vproof)
        .map_err(|e| format!("verify: {e:?}"))
}

/// Regression for upstream 125da1e: the recursive FRI verifier must require at least the
/// configured number of query proofs. The circuit is built from the proof's shape, so without
/// the check a proof whose FRI part carries a single query is verified with a single query.
#[test]
fn recursive_verifier_rejects_too_few_fri_queries() {
    let n: usize = 20;
    let mut builder = CircuitBuilder::new();
    let expected_result = builder.alloc_public_input("expected_result");
    let mut a = builder.alloc_const(F::ZERO, "F(0)");
    let mut b = builder.alloc_const(F::ONE, "F(1)");
    for _i in 2..=n {
        let next = builder.add(a, b);
        a = b;
        b = next;
    }
    builder.connect(b, expected_result);
    let circuit = builder.build().unwrap();

    let table_packing = TablePacking::new(2, 4);
    let config_proving = make_test_config();
    let (airs_degrees, primitive_columns, non_primitive_columns) =
        get_airs_and_degrees_with_prep::<MyConfig, _, 1>(
            &circuit,
            &table_packing,
            &[],
            &[],
            ConstraintProfile::Standard,
        )
        .unwrap();
    let (airs, degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();
    let mut runner = circuit.runner();
    runner
        .set_public_inputs(&[compute_fibonacci_classical(n)])
        .unwrap();
    let traces = runner.run().unwrap();
    let prover_data = ProverData::from_airs_and_degrees(&config_proving, &airs, &degrees);
    let circuit_prover_data =
        CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);
    let prover = BatchStarkProver::new(config_proving).with_table_packing(table_packing);
    let proof = prover
        .prove_all_tables(&traces, &circuit_prover_data)
        .unwrap();
    let common = circuit_prover_data.common_data();

    let scalars = test_fri_scalars();
    let fri_verifier_params = FriVerifierParams::with_mmcs(
        scalars.log_blowup,
        scalars.log_final_poly_len,
        scalars.commit_pow_bits,
        scalars.query_pow_bits,
        scalars.num_queries,
        Poseidon2Config::KOALA_BEAR_D4_W16,
    );

    verify_fib_batch_proof_recursively(&proof, common, &fri_verifier_params)
        .expect("honest proof must verify recursively");

    let mut truncated = proof;
    let q = truncated.proof.opening_proof.query_proofs.len();
    assert!(q > 1, "test FRI config must use more than one query");
    truncated.proof.opening_proof.query_proofs.truncate(1);
    let res = verify_fib_batch_proof_recursively(&truncated, common, &fri_verifier_params);
    println!("single-query proof (honest has {q}): {res:?}");
    assert!(
        res.is_err(),
        "a proof carrying 1 of {q} required FRI queries must be rejected"
    );
}

fn compute_fibonacci_classical(n: usize) -> F {
    if n == 0 {
        return F::ZERO;
    }
    if n == 1 {
        return F::ONE;
    }

    let mut a = F::ZERO;
    let mut b = F::ONE;

    for _i in 2..=n {
        let next = a + b;
        a = b;
        b = next;
    }

    b
}
