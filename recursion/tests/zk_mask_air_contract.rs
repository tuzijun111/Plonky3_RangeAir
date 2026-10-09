//! AIR contract under ZK LogUp masks, and the cost of the masks on a recursion circuit.
//!
//! Under ZK the main window of every AIR with global lookups also holds `D` trailing mask
//! columns per global lookup. An AIR must only read its own `width()` columns, must not assert
//! on the window length, and must not let its constraint set depend on it.
//!
//! Run these tests in a debug build as well (`cargo test -p p3-recursion --test
//! zk_mask_air_contract`): the debug prover evaluates every AIR on the masked trace
//! (`check_constraints`), which trips the width assertions of an AIR that breaks the contract
//! (before the fix: the ALU `debug_assert_eq!` on the window length and the Poseidon2 row
//! `Borrow` suffix assertion).

mod common;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use p3_air::BaseAir;
use p3_air::symbolic::{AirLayout, BaseEntry, BaseLeaf, SymbolicExpr, SymbolicExpression};
use p3_batch_stark::ProverData;
use p3_batch_stark::symbolic::get_symbolic_constraints;
use p3_circuit::ops::{generate_poseidon2_trace, generate_recompose_trace};
use p3_circuit::{Circuit, CircuitBuilder, Traces};
use p3_circuit_prover::batch_stark_prover::{poseidon2_air_builders, recompose_air_builders};
use p3_circuit_prover::common::{NpoPreprocessor, get_airs_and_degrees_with_prep};
use p3_circuit_prover::{
    BatchStarkProver, CircuitProverData, ConstraintProfile, DEFAULT_ZK_MIN_TRACE_HEIGHT,
    Poseidon2Preprocessor, RecomposePreprocessor, TablePacking,
};
use p3_fri::{FriParameters, HidingFriPcs};
use p3_lookup::logup::LogUpGadget;
use p3_lookup::lookup_mask_width;
use p3_poseidon2_circuit_air::KoalaBearD4Width16;
use p3_recursion::Poseidon2Config;
use p3_recursion::pcs::fri::{FriVerifierParams, InputProofTargets, MerkleCapTargets, RecValMmcs};
use p3_recursion::pcs::set_fri_mmcs_private_data;
use p3_recursion::verifier::verify_p3_batch_proof_circuit;
use p3_test_utils::koala_bear_params::*;
use p3_test_utils::test_fri_scalars;
use p3_uni_stark::StarkGenericConfig;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use crate::common::InnerFriGeneric;

type InnerFri = InnerFriGeneric<MyConfig, MyHash, MyCompress, DIGEST_ELEMS>;
/// Test-only ZK config (seeded hiding RNG; never for real proofs).
type MyPcsZk = HidingFriPcs<F, Dft, MyMmcs, ChallengeMmcs, SmallRng>;
type MyConfigZk = StarkConfig<MyPcsZk, Challenge, Challenger>;

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

/// A non-ZK Fibonacci proof and the traces of its in-circuit verifier: a D=4 circuit with
/// Const, Public, ALU (with packed Horner columns), Poseidon2 and recompose tables.
fn verifier_circuit_and_traces() -> (Circuit<Challenge>, Traces<Challenge>) {
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
    let table_packing = TablePacking::new(2, 4);
    let config = make_test_config();
    let circuit = builder.build().unwrap();
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
    let mut x = (F::ZERO, F::ONE);
    for _ in 2..=n {
        x = (x.1, x.0 + x.1);
    }
    runner.set_public_inputs(&[x.1]).unwrap();
    let traces = runner.run().unwrap();
    let prover_data = ProverData::from_airs_and_degrees(&config, &airs, &degrees);
    let cpd = CircuitProverData::new(prover_data, pc, npc);
    let prover = BatchStarkProver::new(config).with_table_packing(table_packing);
    let proof = prover.prove_all_tables(&traces, &cpd).unwrap();
    let common = cpd.common_data();

    let s = test_fri_scalars();
    let fvp = FriVerifierParams::with_mmcs(
        s.log_blowup,
        s.log_final_poly_len,
        s.commit_pow_bits,
        s.query_pow_bits,
        s.num_queries,
        POSEIDON2,
    );
    let num_tables = common
        .preprocessed
        .as_ref()
        .map_or(0, |g| g.instances.len());
    let pis: Vec<Vec<F>> = vec![vec![]; num_tables];
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
        &make_test_config(),
        &mut cb,
        &proof,
        &fvp,
        common,
        &LogUpGadget::new(),
        POSEIDON2,
        &[],
    )
    .unwrap();
    let vc = cb.build().unwrap();
    let (pubi, privi) = vi.pack_values(&pis, &proof.proof, common);
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
    let traces = r.run().unwrap();
    (vc, traces)
}

/// AIRs, prover data and table packing of the verifier circuit, for config `SC`.
macro_rules! verifier_airs {
    ($sc:ty, $config:expr, $circuit:expr, $packing:expr) => {{
        let npo_prep: Vec<Box<dyn NpoPreprocessor<F>>> = vec![
            Box::new(Poseidon2Preprocessor),
            Box::new(RecomposePreprocessor::new(true)),
        ];
        let mut air_builders = poseidon2_air_builders::<_, 4>();
        air_builders.extend(recompose_air_builders(1, true));
        let (airs_degrees, pc, npc) = get_airs_and_degrees_with_prep::<$sc, _, 4>(
            $circuit,
            $packing,
            &npo_prep,
            &air_builders,
            ConstraintProfile::Standard,
        )
        .unwrap();
        let (airs, degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();
        let ext: Vec<usize> = degrees.iter().map(|&d| d + $config.is_zk()).collect();
        let prover_data = ProverData::from_airs_and_degrees(&$config, &airs, &ext);
        (airs, CircuitProverData::new(prover_data, pc, npc))
    }};
}

/// The debug-build prover runs `check_constraints` on the masked trace of every table here
/// (Const, Public, ALU with packed Horner, Poseidon2, recompose), so in a debug build this is
/// the AIR-contract check; in release it is a completeness check and a cost probe.
#[test]
fn zk_masked_proof_of_npo_tables_respects_air_contract() {
    let (vc, vtraces) = verifier_circuit_and_traces();

    let packing = TablePacking::new(1, 8).with_min_trace_height(DEFAULT_ZK_MIN_TRACE_HEIGHT);
    let cz = zk_config(7);
    let (vairs, vcpd) = verifier_airs!(MyConfigZk, cz, &vc, &packing);
    let air_widths: Vec<usize> = vairs.iter().map(BaseAir::<F>::width).collect();
    let mut vp = BatchStarkProver::new(cz).with_table_packing(packing.clone());
    vp.register_poseidon2_table::<4>(POSEIDON2);
    vp.register_recompose_table::<4>(true);

    let reps = if cfg!(debug_assertions) { 1 } else { 5 };
    let mut times = Vec::with_capacity(reps);
    let mut last = None;
    for _ in 0..reps {
        let t = Instant::now();
        let p = vp.prove_all_tables(&vtraces, &vcpd).unwrap();
        times.push(t.elapsed().as_secs_f64() * 1e3);
        last = Some(p);
    }
    let p = last.unwrap();
    vp.verify_all_tables(&p)
        .expect("native verification of the masked proof");
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let widths: Vec<usize> = p
        .proof
        .opened_values
        .instances
        .iter()
        .map(|o| o.base_opened_values.trace_local.len())
        .collect();
    let expected: Vec<usize> = air_widths
        .iter()
        .zip(vcpd.common_data().lookups.iter())
        .map(|(w, l)| w + lookup_mask_width(l, D, true))
        .collect();
    assert_eq!(widths, expected);
    let globals: Vec<usize> = p.proof.global_lookup_data.iter().map(Vec::len).collect();
    let bytes = postcard::to_allocvec(&p.proof).unwrap().len();
    println!(
        "ZK-masked: AIR widths {air_widths:?} -> main widths {widths:?}; globals {globals:?} \
         (sum {}); prove ms median of {reps}: {:.1}; batch proof bytes {bytes}",
        globals.iter().sum::<usize>(),
        times[times.len() / 2]
    );

    // Reference (not a masked-vs-unmasked comparison): the same circuit and packing proved
    // without ZK, which has neither the hiding randomization nor the masks.
    if !cfg!(debug_assertions) {
        let c = make_test_config();
        let (airs, cpd) = verifier_airs!(MyConfig, c, &vc, &packing);
        let mut prover = BatchStarkProver::new(c).with_table_packing(packing);
        prover.register_poseidon2_table::<4>(POSEIDON2);
        prover.register_recompose_table::<4>(true);
        let mut t = Vec::with_capacity(reps);
        let mut last = None;
        for _ in 0..reps {
            let s = Instant::now();
            last = Some(prover.prove_all_tables(&vtraces, &cpd).unwrap());
            t.push(s.elapsed().as_secs_f64() * 1e3);
        }
        let q = last.unwrap();
        prover.verify_all_tables(&q).unwrap();
        t.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let w: Vec<usize> = q
            .proof
            .opened_values
            .instances
            .iter()
            .map(|o| o.base_opened_values.trace_local.len())
            .collect();
        assert_eq!(w, airs.iter().map(BaseAir::<F>::width).collect::<Vec<_>>());
        println!(
            "non-ZK reference: main widths {w:?}; prove ms median of {reps}: {:.1}; batch proof \
             bytes {}",
            t[t.len() / 2],
            postcard::to_allocvec(&q.proof).unwrap().len()
        );
    }
}

/// Evaluate a base-field symbolic expression at a point, memoizing shared sub-expressions.
fn eval_at(
    e: &SymbolicExpression<F>,
    point: &mut HashMap<(BaseEntry, usize), F>,
    rng: &mut SmallRng,
    selectors: [F; 3],
    memo: &mut HashMap<*const SymbolicExpression<F>, F>,
) -> F {
    let child = |x: &Arc<SymbolicExpression<F>>,
                 point: &mut HashMap<(BaseEntry, usize), F>,
                 rng: &mut SmallRng,
                 memo: &mut HashMap<*const SymbolicExpression<F>, F>| {
        let key = Arc::as_ptr(x);
        if let Some(v) = memo.get(&key) {
            return *v;
        }
        let v = eval_at(x, point, rng, selectors, memo);
        memo.insert(key, v);
        v
    };
    match e {
        SymbolicExpr::Leaf(BaseLeaf::Variable(v)) => *point
            .entry((v.entry, v.index))
            .or_insert_with(|| rng.random()),
        SymbolicExpr::Leaf(BaseLeaf::IsFirstRow) => selectors[0],
        SymbolicExpr::Leaf(BaseLeaf::IsLastRow) => selectors[1],
        SymbolicExpr::Leaf(BaseLeaf::IsTransition) => selectors[2],
        SymbolicExpr::Leaf(BaseLeaf::Constant(c)) => *c,
        SymbolicExpr::Add { x, y, .. } => child(x, point, rng, memo) + child(y, point, rng, memo),
        SymbolicExpr::Sub { x, y, .. } => child(x, point, rng, memo) - child(y, point, rng, memo),
        SymbolicExpr::Neg { x, .. } => -child(x, point, rng, memo),
        SymbolicExpr::Mul { x, y, .. } => child(x, point, rng, memo) * child(y, point, rng, memo),
    }
}

/// Every table AIR of a recursion circuit has the same constraint polynomials whether its
/// main window is exactly `width()` or carries trailing mask columns: same number, same
/// degrees, and equal values at a random point (variables beyond `width()` would get fresh
/// random values and change them).
#[test]
fn air_constraints_do_not_depend_on_main_window_width() {
    let (vc, _) = verifier_circuit_and_traces();
    let packing = TablePacking::new(1, 8).with_min_trace_height(DEFAULT_ZK_MIN_TRACE_HEIGHT);
    let cz = zk_config(8);
    let (airs, cpd) = verifier_airs!(MyConfigZk, cz, &vc, &packing);
    let lookups = &cpd.common_data().lookups;
    let gadget = LogUpGadget::new();
    let mut rng = SmallRng::seed_from_u64(0x5eed);
    let selectors: [F; 3] = [rng.random(), rng.random(), rng.random()];
    let mut checked_constraints = 0;
    assert!(
        airs.iter()
            .filter(|a| BaseAir::<F>::width(*a) >= 100)
            .count()
            >= 2,
        "the fixture must contain the ALU (packed Horner) and Poseidon2 tables"
    );

    for (i, air) in airs.iter().enumerate() {
        let w = BaseAir::<F>::width(air);
        // The real mask width of this table, or a stand-in if it has no global lookups.
        let extra = lookup_mask_width(&lookups[i], D, true).max(2 * D);
        let layout = |main_width| AirLayout {
            preprocessed_width: BaseAir::<F>::preprocessed_width(air),
            main_width,
            num_public_values: BaseAir::<F>::num_public_values(air),
            ..Default::default()
        };
        // AIR constraints only (no lookup contexts): the window width must not matter.
        let (base_plain, ext_plain) =
            get_symbolic_constraints::<F, Challenge, _, _>(air, layout(w), &[], &gadget);
        let (base_wide, ext_wide) =
            get_symbolic_constraints::<F, Challenge, _, _>(air, layout(w + extra), &[], &gadget);
        assert_eq!(
            base_plain.len(),
            base_wide.len(),
            "table {i}: base constraint count"
        );
        assert_eq!(
            ext_plain.len(),
            ext_wide.len(),
            "table {i}: ext constraint count"
        );
        // (Some tables, e.g. Public, have no AIR constraints of their own, only lookups.)
        checked_constraints += base_plain.len();
        if w >= 100 {
            // The wide tables (ALU with packed Horner, Poseidon2) must actually be exercised.
            assert!(!base_plain.is_empty(), "table {i}: no AIR constraints");
        }

        let mut point = HashMap::new();
        let mut memo_plain = HashMap::new();
        let mut memo_wide = HashMap::new();
        for (k, (cp, cw)) in base_plain.iter().zip(&base_wide).enumerate() {
            assert_eq!(
                cp.degree_multiple(),
                cw.degree_multiple(),
                "table {i}: constraint {k} degree"
            );
            let vp = eval_at(cp, &mut point, &mut rng, selectors, &mut memo_plain);
            let vw = eval_at(cw, &mut point, &mut rng, selectors, &mut memo_wide);
            assert_eq!(
                vp, vw,
                "table {i}: constraint {k} differs with a wider window"
            );
        }
        assert!(
            point
                .keys()
                .all(|(e, idx)| !matches!(e, BaseEntry::Main { .. }) || *idx < w),
            "table {i}: an AIR constraint reads a main column beyond width() = {w}"
        );
        println!(
            "table {i}: width {w} (+{extra}), {} base constraints identical",
            base_plain.len()
        );
    }
    assert!(checked_constraints > 0);
}
