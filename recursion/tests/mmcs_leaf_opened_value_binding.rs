//! End-to-end regression for the MMCS leaf-hash coefficient binding (upstream 98b835f),
//! BabyBear D=4 width-16 (zkSketch's configuration).
//!
//! `verify_batch_circuit` packs a leaf's opened base-field values into extension limbs before
//! hashing. Packed through the ALU `mul_add` chain, a limb is tied to its opened values only by
//! `sum(c_i * X^i)`, which leaves each opened value `D - 1` free base dimensions: a prover can
//! claim `o0' = o0 + delta * X` (not a base-field element) and `o1' = o1 - delta`, keep every
//! limb (and so the leaf digest and the Merkle root) unchanged, and hand the verifier opened
//! values of its choosing for the rest of the circuit to compute with. Packed through
//! `recompose/coeff`, each opened value must equal the base-field coefficient the limb carries.

use std::panic::AssertUnwindSafe;

use itertools::Itertools;
use p3_batch_stark::ProverData;
use p3_circuit::ops::{
    Poseidon2Config, generate_poseidon2_trace, generate_recompose_trace, perm_private_data,
};
use p3_circuit::{CircuitBuilder, CircuitRunner};
use p3_circuit_prover::batch_stark_prover::{poseidon2_air_builders, recompose_air_builders};
use p3_circuit_prover::common::{NpoPreprocessor, get_airs_and_degrees_with_prep};
use p3_circuit_prover::config::BabyBearConfig;
use p3_circuit_prover::{
    BatchStarkProver, CircuitProverData, ConstraintProfile, Poseidon2Preprocessor,
    RecomposePreprocessor, TablePacking, config,
};
use p3_commit::Mmcs;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_poseidon2_circuit_air::BabyBearD4Width16;
use p3_recursion::Target;
use p3_recursion::pcs::verify_batch_circuit;
use p3_test_utils::baby_bear_params::*;

type EF = Challenge;
const CFG: Poseidon2Config = Poseidon2Config::BABY_BEAR_D4_W16;

/// Build the leaf-hash circuit, run it with `opened` as the claimed opened values, then prove and
/// verify. `Err` = rejected (witness generation or proof).
fn run_leaf_hash(opened_override: Option<Vec<EF>>) -> Result<(), String> {
    let perm = default_babybear_poseidon2_16();
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let mmcs = MyMmcs::new(hash, compress, 0);

    // One 2x8 matrix: a leaf is exactly one sponge rate (two D=4 limbs).
    let mat = RowMajorMatrix::new((1..=16).map(F::from_u32).collect(), 8);
    let dimensions = vec![mat.dimensions()];
    let (commit, prover_data) = mmcs.commit(vec![mat]);
    let index = 1usize;
    let opening = mmcs.open_batch(index, &prover_data);

    let mut builder = CircuitBuilder::<EF>::new();
    builder.enable_poseidon2_perm::<BabyBearD4Width16, _>(
        generate_poseidon2_trace::<EF, BabyBearD4Width16>,
        perm,
    );
    builder.enable_recompose::<F>(generate_recompose_trace::<F, EF>);

    let opened: Vec<Vec<Target>> = opening
        .opened_values
        .iter()
        .map(|v| v.iter().map(|_| builder.public_input()).collect_vec())
        .collect();
    // The rest of a verifier computes with the opened values.
    let _ = builder.mul(opened[0][0], opened[0][1]);
    let directions = builder.alloc_public_inputs(1, "directions");
    let mut cap = Vec::new();
    for _ in 0..commit.num_roots() {
        let lifted: Vec<Target> = (0..CFG.rate()).map(|_| builder.public_input()).collect();
        let packed: Vec<Target> = lifted
            .chunks(D)
            .map(|c| builder.recompose_base_coeffs_to_ext_via_alu::<F>(c).unwrap())
            .collect();
        cap.push(packed);
    }
    let op_ids = verify_batch_circuit::<F, EF>(
        &mut builder,
        CFG,
        &cap,
        &dimensions,
        &directions,
        &opened,
        None,
    )
    .map_err(|e| format!("build: {e:?}"))?;
    let circuit = builder.build().map_err(|e| format!("build: {e:?}"))?;

    let honest_opened: Vec<EF> = opening.opened_values[0]
        .iter()
        .map(|&v| EF::from(v))
        .collect();
    let mut publics = opened_override.unwrap_or(honest_opened);
    publics.push(EF::from_bool(index & 1 == 1));
    for root in commit.roots() {
        publics.extend(root.iter().map(|&v| EF::from(v)));
    }

    let set_private = |runner: &mut CircuitRunner<'_, EF>| {
        let siblings = opening
            .opening_proof
            .iter()
            .map(|d| {
                d.chunks(D)
                    .map(EF::from_basis_coefficients_slice)
                    .collect::<Option<Vec<_>>>()
                    .unwrap()
            })
            .collect_vec();
        for (&op_id, sib) in op_ids.iter().zip(siblings) {
            runner
                .set_private_data(op_id, perm_private_data(CFG, sib))
                .unwrap();
        }
    };
    let mut runner = circuit.runner();
    runner.set_public_inputs(&publics).unwrap();
    set_private(&mut runner);
    let traces = runner
        .run()
        .map_err(|e| format!("witness generation: {e:?}"))?;

    let table_packing = TablePacking::new(1, 1);
    let stark_config = config::baby_bear();
    let npo: Vec<Box<dyn NpoPreprocessor<F>>> = vec![
        Box::new(Poseidon2Preprocessor),
        Box::new(RecomposePreprocessor::new(true)),
    ];
    let mut air_builders = poseidon2_air_builders::<BabyBearConfig, D>();
    air_builders.extend(recompose_air_builders::<BabyBearConfig, D>(1, true));
    let (ad, pc, npc) = get_airs_and_degrees_with_prep::<BabyBearConfig, EF, D>(
        &circuit,
        &table_packing,
        &npo,
        &air_builders,
        ConstraintProfile::Standard,
    )
    .map_err(|e| format!("airs: {e:?}"))?;
    let (airs, degs): (Vec<_>, Vec<usize>) = ad.into_iter().unzip();
    let data = CircuitProverData::new(
        ProverData::from_airs_and_degrees(&stark_config, &airs, &degs),
        pc,
        npc,
    );
    let mut prover = BatchStarkProver::new(stark_config).with_table_packing(table_packing);
    prover.register_poseidon2_table::<D>(CFG);
    prover.register_recompose_table::<D>(true);
    std::panic::catch_unwind(AssertUnwindSafe(|| {
        let proof = prover
            .prove_all_tables(&traces, &data)
            .map_err(|e| format!("prove: {e:?}"))?;
        prover
            .verify_all_tables(&proof)
            .map_err(|e| format!("verify: {e:?}"))
    }))
    .unwrap_or_else(|_| Err("prover panicked".to_string()))
}

#[test]
fn honest_leaf_opening_verifies() {
    run_leaf_hash(None).expect("honest leaf opening must verify");
}

#[test]
fn non_base_shifted_opened_values_are_rejected() {
    let perm = default_babybear_poseidon2_16();
    let mmcs = MyMmcs::new(MyHash::new(perm.clone()), MyCompress::new(perm), 0);
    let mat = RowMajorMatrix::new((1..=16).map(F::from_u32).collect(), 8);
    let (_, pd) = mmcs.commit(vec![mat]);
    let honest: Vec<F> = mmcs.open_batch(1, &pd).opened_values[0].clone();

    // o0' = o0 + delta * X, o1' = o1 - delta: the packed limb o0' + o1' * X is unchanged.
    let delta = F::from_u32(7);
    let mut forged: Vec<EF> = honest.iter().map(|&v| EF::from(v)).collect();
    forged[0] = EF::from_basis_coefficients_fn(|j| match j {
        0 => honest[0],
        1 => delta,
        _ => F::ZERO,
    });
    forged[1] = EF::from(honest[1] - delta);
    let outcome = run_leaf_hash(Some(forged));
    println!("non-base shifted opening: {outcome:?}");
    assert!(
        outcome.is_err(),
        "opened values that are not the leaf's base-field coefficients must be rejected"
    );
}
