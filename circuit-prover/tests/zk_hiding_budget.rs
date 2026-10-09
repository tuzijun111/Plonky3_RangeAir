//! Hiding-budget guard: under a ZK PCS, `BatchStarkProver` refuses table packings whose minimum
//! trace height is below `TablePacking::zk_min_trace_height` (default 128), since shorter tables
//! may not hide their LogUp masks and auxiliary columns. The guard does not apply without ZK,
//! and the setting is prover-side only (not serialized into proofs).

use p3_batch_stark::ProverData;
use p3_circuit::CircuitBuilder;
use p3_circuit_prover::common::get_airs_and_degrees_with_prep;
use p3_circuit_prover::{
    BatchStarkProver, BatchStarkProverError, CircuitProverData, ConstraintProfile,
    DEFAULT_ZK_MIN_TRACE_HEIGHT, TablePacking,
};
use p3_fri::HidingFriPcs;
use p3_test_utils::koala_bear_params::*;
use p3_uni_stark::StarkGenericConfig;
use rand::SeedableRng;
use rand::rngs::SmallRng;

/// Test-only ZK config (seeded hiding RNG; never for real proofs).
type MyPcsZk = HidingFriPcs<F, Dft, MyMmcs, ChallengeMmcs, SmallRng>;
type MyConfigZk = StarkConfig<MyPcsZk, Challenge, Challenger>;

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

/// Prove a small Fibonacci circuit with the given packing; verify it when proving succeeds.
macro_rules! prove_fib {
    ($sc:ty, $config:expr, $packing:expr) => {{
        let n = 20usize;
        let mut builder = CircuitBuilder::new();
        let expected = builder.alloc_public_input("expected");
        let mut a = builder.alloc_const(F::ZERO, "F(0)");
        let mut b = builder.alloc_const(F::ONE, "F(1)");
        for _ in 2..=n {
            let next = builder.add(a, b);
            a = b;
            b = next;
        }
        builder.connect(b, expected);
        let mut x = (F::ZERO, F::ONE);
        for _ in 2..=n {
            x = (x.1, x.0 + x.1);
        }
        let circuit = builder.build().unwrap();
        let config = $config;
        let packing: TablePacking = $packing;
        let (airs_degrees, pc, npc) = get_airs_and_degrees_with_prep::<$sc, _, 1>(
            &circuit,
            &packing,
            &[],
            &[],
            ConstraintProfile::Standard,
        )
        .unwrap();
        let (airs, degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();
        let ext: Vec<usize> = degrees.iter().map(|&d| d + config.is_zk()).collect();
        let mut runner = circuit.runner();
        runner.set_public_inputs(&[x.1]).unwrap();
        let traces = runner.run().unwrap();
        let prover_data = ProverData::from_airs_and_degrees(&config, &airs, &ext);
        let cpd = CircuitProverData::new(prover_data, pc, npc);
        let prover = BatchStarkProver::new(config).with_table_packing(packing);
        let res = prover.prove_all_tables(&traces, &cpd);
        if let Ok(proof) = &res {
            prover.verify_all_tables(proof).expect("verify");
        }
        res
    }};
}

#[test]
fn zk_prover_rejects_tables_below_hiding_budget() {
    for (min, seed) in [(1usize, 1u64), (32, 2), (64, 3)] {
        let res = prove_fib!(
            MyConfigZk,
            zk_config(seed),
            TablePacking::new(1, 2).with_min_trace_height(min)
        );
        match res {
            Err(BatchStarkProverError::ZkTraceHeightBelowHidingBudget {
                min_trace_height,
                required,
            }) => {
                assert_eq!(min_trace_height, min);
                assert_eq!(required, DEFAULT_ZK_MIN_TRACE_HEIGHT);
            }
            Err(e) => panic!("min height {min}: unexpected error {e:?}"),
            Ok(_) => panic!("min height {min}: ZK proof below the hiding budget was produced"),
        }
    }
}

#[test]
fn zk_prover_accepts_tables_at_or_above_hiding_budget() {
    assert_eq!(DEFAULT_ZK_MIN_TRACE_HEIGHT, 128);
    for (min, seed) in [(128usize, 4u64), (512, 5)] {
        prove_fib!(
            MyConfigZk,
            zk_config(seed),
            TablePacking::new(1, 2).with_min_trace_height(min)
        )
        .unwrap_or_else(|e| panic!("min height {min}: {e:?}"));
    }
    // zkSketch's packing: with_fri_params(6, 2) gives 2^(6 + 2 + 1) = 512 rows.
    let p = TablePacking::new(1, 2).with_fri_params(6, 2);
    assert_eq!(p.min_trace_height(), 512);
    assert!(p.min_trace_height() >= p.zk_min_trace_height());
}

#[test]
fn zk_hiding_budget_can_be_lowered_explicitly() {
    // Test FRI parameters reveal far fewer values (2 queries): 2 * (2 + 4 * 2) = 20 -> 32.
    let budget = TablePacking::zk_hiding_budget(2, D, 2);
    assert_eq!(budget, 32);
    prove_fib!(
        MyConfigZk,
        zk_config(6),
        TablePacking::new(1, 2)
            .with_min_trace_height(budget)
            .with_zk_min_trace_height(budget)
    )
    .expect("explicitly lowered budget");
}

#[test]
fn nonzk_prover_is_not_subject_to_hiding_budget() {
    for min in [1usize, 64] {
        prove_fib!(
            MyConfig,
            make_test_config(),
            TablePacking::new(1, 2).with_min_trace_height(min)
        )
        .unwrap_or_else(|e| panic!("non-ZK, min height {min}: {e:?}"));
    }
}

#[test]
fn zk_hiding_budget_formula() {
    // 2 * (54 + 4 * 2) = 124 -> 128: the default, for 54 queries over a degree-4 extension.
    assert_eq!(TablePacking::zk_hiding_budget(54, 4, 2), 128);
    assert_eq!(TablePacking::zk_hiding_budget(100, 4, 2), 256);
    assert_eq!(TablePacking::zk_hiding_budget(54, 2, 2), 128);
    assert_eq!(TablePacking::new(1, 1).zk_min_trace_height(), 128);
}

#[test]
fn zk_min_trace_height_is_not_serialized_and_ignored_by_equality() {
    let plain = TablePacking::new(2, 4).with_fri_params(6, 2);
    let custom = plain.clone().with_zk_min_trace_height(1024);
    assert_eq!(plain, custom);
    let bytes_plain = postcard::to_allocvec(&plain).unwrap();
    let bytes_custom = postcard::to_allocvec(&custom).unwrap();
    assert_eq!(
        bytes_plain, bytes_custom,
        "the prover-side budget must not change proof bytes"
    );
    let back: TablePacking = postcard::from_bytes(&bytes_custom).unwrap();
    assert_eq!(back, plain);
    assert_eq!(back.zk_min_trace_height(), DEFAULT_ZK_MIN_TRACE_HEIGHT);
}
