//! Zero-knowledge LogUp masks on the NPO tables zkSketch depends on, over several buses.
//!
//! Fixtures (see `zk_mask_common`): a D=1 circuit with Permute (PermuteBus), byte range checks
//! (RangeBus), ALU work and an Expose with public values, all also on WitnessChecks; and a D=4
//! circuit with Poseidon1, Permute, `add_range_bits` and Expose.
//!
//! Honest masked proofs verify natively and in circuit. The forgeries are built with the
//! test-only mask hook (`BatchStarkProver::prove_all_tables_with_mask_hook`), so the prover
//! itself absorbs the forged revealed sums into the transcript: the challenges are consistent
//! with what the proof reveals, and only the masked last-row LogUp binding
//! `(c' - rho - s[n-1]) * den - num = 0` or the per-bus sum check can catch them.

mod zk_mask_common;

use p3_circuit_prover::LookupMaskPlan;
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_test_utils::koala_bear_params::*;
use zk_mask_common::*;

fn delta() -> Challenge {
    Challenge::from_u64(5)
}

fn revealed(p: &ZkProved, (i, g): (usize, usize)) -> Challenge {
    p.proof.proof.global_lookup_data[i][g].cumulative_sum
}

/// Native verification must fail with an error whose debug form contains `needle`.
fn assert_native_rejects<EF, const TD: usize>(fx: &NpoFixture<EF, TD>, p: &ZkProved, needle: &str)
where
    EF: p3_field::Field
        + p3_field::ExtensionField<F>
        + p3_circuit_prover::field_params::ExtractBinomialW<F>,
{
    match fx.verify_native(p) {
        Err(e) => {
            let msg = format!("{e:?}");
            println!("native: {msg}");
            assert!(msg.contains(needle), "expected {needle}, got {msg}");
        }
        Ok(()) => panic!("native verifier accepted a forged proof"),
    }
}

/// Which in-circuit check is expected to reject a forgery.
#[derive(Clone, Copy, Debug)]
enum Check {
    /// The masked last-row LogUp binding, inside the out-of-domain quotient identity.
    Binding,
    /// The per-bus sum check: some bus total is `+delta` or `-delta` instead of zero. (Only
    /// release builds reach it: the debug prover refuses to finish such proofs.)
    #[cfg_attr(debug_assertions, allow(dead_code))]
    PerBus,
}

/// The in-circuit verifier must build (the proof shape is honest) and then fail to run, in the
/// expected check. A failing per-bus check is a conflict between the bus total and zero, so its
/// message carries the total, `+delta` or `-delta`; the quotient identity of a binding forgery
/// compares two unrelated extension-field values instead.
fn assert_in_circuit_rejects<EF, const TD: usize>(
    fx: &NpoFixture<EF, TD>,
    p: &ZkProved,
    check: Check,
) where
    EF: p3_field::Field
        + p3_field::ExtensionField<F>
        + p3_circuit_prover::field_params::ExtractBinomialW<F>,
{
    let built = build_verifier::<TD>(p, &(fx.table_provers)())
        .expect("a forged proof with an honest shape still yields a verifier circuit");
    let err = run_verifier(&built, &p.proof).expect_err("in-circuit verifier accepted a forgery");
    println!("in-circuit ({check:?}): {err}");
    assert!(err.contains("WitnessConflict"), "unexpected error: {err}");
    let bus_total = |d: Challenge| {
        format!(
            "value: {:?}",
            BasedVectorSpace::<F>::as_basis_coefficients_slice(&d)
        )
    };
    let names_delta = err.contains(&bus_total(delta())) || err.contains(&bus_total(-delta()));
    match check {
        Check::PerBus => assert!(names_delta, "expected a per-bus conflict: {err}"),
        Check::Binding => assert!(!names_delta, "expected a binding conflict: {err}"),
    }
}

/// Honest masked proof of `fx`: masked widths, per-bus balance, and acceptance natively and in
/// circuit. Returns the proof.
fn honest<EF, const TD: usize>(fx: &NpoFixture<EF, TD>, seed: u64) -> ZkProved
where
    EF: p3_field::Field
        + p3_field::ExtensionField<F>
        + p3_circuit_prover::field_params::ExtractBinomialW<F>,
{
    let p = fx.prove(seed, None);
    let widths = opened_main_widths(&p);
    assert_eq!(widths, expected_masked_widths(&p));
    let ops: Vec<String> = p
        .proof
        .non_primitives
        .iter()
        .map(|e| format!("{:?}", e.op_type))
        .collect();
    println!(
        "{}: NPO tables {ops:?}; buses {:?}; AIR widths {:?} -> masked {widths:?}",
        fx.name,
        bus_sizes(&p.cpd),
        p.air_widths
    );
    for (bus, sum) in per_bus_sums(&p) {
        assert_eq!(sum, Challenge::ZERO, "bus {bus} does not balance");
    }
    assert!(
        p.proof
            .non_primitives
            .iter()
            .any(|e| !e.public_values.is_empty()),
        "the Expose table must carry public values"
    );
    fx.verify_native(&p)
        .expect("native verification of an honest masked proof");
    fx.verify_in_circuit(&p)
        .expect("in-circuit verification of an honest masked proof");
    p
}

#[test]
fn zk_npo_multibus_d1_honest_verifies_natively_and_in_circuit() {
    let fx = multibus_d1();
    let p = honest(&fx, 1);
    let buses = bus_sizes(&p.cpd);
    for bus in ["PermuteBus", "RangeBus", "WitnessChecks"] {
        assert!(
            buses.get(bus).copied().unwrap_or(0) >= 2,
            "{bus} must have several members: {buses:?}"
        );
    }
    // NPO tables carry masks: every table with a global lookup is wider than its AIR.
    let widths = opened_main_widths(&p);
    assert!(
        widths[3..]
            .iter()
            .zip(&p.air_widths[3..])
            .all(|(w, a)| w > a)
    );
}

#[test]
fn zk_npo_d4_poseidon1_range_bits_honest_verifies_natively_and_in_circuit() {
    let fx = npo_d4();
    let p = honest(&fx, 2);
    let ops: Vec<String> = p
        .proof
        .non_primitives
        .iter()
        .map(|e| format!("{:?}", e.op_type))
        .collect();
    for needle in ["poseidon1", "permute", "range", "expose"] {
        assert!(
            ops.iter().any(|o| o.to_lowercase().contains(needle)),
            "missing {needle} table: {ops:?}"
        );
    }
    let buses = bus_sizes(&p.cpd);
    for bus in ["PermuteBus", "RangeBus"] {
        assert!(buses.get(bus).copied().unwrap_or(0) >= 2, "{buses:?}");
    }
}

/// The verifier circuit of an NPO proof does not depend on the masks.
#[test]
fn zk_npo_verifier_circuit_is_mask_independent() {
    let fx = multibus_d1();
    let a = fx.prove(21, None);
    let b = fx.prove(22, None);
    assert!(
        revealed_sums(&a)
            .iter()
            .zip(revealed_sums(&b))
            .all(|(x, y)| *x != y),
        "masks must be fresh per proof"
    );
    let tp = (fx.table_provers)();
    let va = build_verifier::<1>(&a, &tp).unwrap();
    let vb = build_verifier::<1>(&b, &tp).unwrap();
    assert_eq!(va.circuit.ops, vb.circuit.ops);
    run_verifier(&va, &a.proof).unwrap();
    run_verifier(&vb, &b.proof).unwrap();
}

/// Same-bus revealed-sum shift, transcript-consistent: commit the honest masks but reveal
/// `+delta` on one member of `bus` and `-delta` on another. The bus still balances and the
/// challenges are those of the forged transcript, so only the masked last-row binding catches it
/// (natively an OOD mismatch).
fn same_bus_reveal_shift<EF, const TD: usize>(fx: &NpoFixture<EF, TD>, bus: &str, seed: u64)
where
    EF: p3_field::Field
        + p3_field::ExtensionField<F>
        + p3_circuit_prover::field_params::ExtractBinomialW<F>,
{
    let setup = fx.setup(seed);
    let [a, b] = two_members(&setup.cpd, bus);
    let honest = fx.prove(seed, None);
    let mut hook = |plan: &mut LookupMaskPlan<Challenge>| {
        plan.reveal_offsets[a.0][a.1] += delta();
        plan.reveal_offsets[b.0][b.1] -= delta();
    };
    let forged = fx.prove(seed, Some(&mut hook));
    // Same seed, same masks: exactly the two revealed sums moved.
    assert_eq!(revealed(&forged, a), revealed(&honest, a) + delta());
    assert_eq!(revealed(&forged, b), revealed(&honest, b) - delta());
    assert_eq!(per_bus_sums(&forged), per_bus_sums(&honest));
    assert_native_rejects(fx, &forged, "OodEvaluationMismatch");
    assert_in_circuit_rejects(fx, &forged, Check::Binding);
}

#[test]
fn zk_npo_same_bus_reveal_shift_rejected_permute_bus() {
    same_bus_reveal_shift(&multibus_d1(), "PermuteBus", 31);
}

#[test]
fn zk_npo_same_bus_reveal_shift_rejected_range_bus() {
    same_bus_reveal_shift(&multibus_d1(), "RangeBus", 32);
}

#[test]
fn zk_npo_same_bus_reveal_shift_rejected_d4_range_bits() {
    same_bus_reveal_shift(&npo_d4(), "RangeBus", 33);
}

/// Mask `+delta` with revealed `-delta` on one member: the revealed sum is the honest one (so
/// every bus balances and the transcript is the honest transcript) but the committed mask no
/// longer matches it, which only the masked last-row binding sees.
#[test]
fn zk_npo_mask_and_reveal_cancel_rejected() {
    let fx = multibus_d1();
    let seed = 41;
    let [a, _] = two_members(&fx.setup(seed).cpd, "RangeBus");
    let mut hook = |plan: &mut LookupMaskPlan<Challenge>| {
        plan.committed[a.0][a.1] += delta();
        plan.reveal_offsets[a.0][a.1] -= delta();
    };
    let forged = fx.prove(seed, Some(&mut hook));
    // (The committed masks enter the main commitment, so the lookup challenges, and with them
    // the true sums, differ from those of an honest proof; every bus still balances.)
    for (bus, sum) in per_bus_sums(&forged) {
        assert_eq!(sum, Challenge::ZERO, "bus {bus} must still balance");
    }
    assert_native_rejects(&fx, &forged, "OodEvaluationMismatch");
    assert_in_circuit_rejects(&fx, &forged, Check::Binding);
}

/// Zero-sum re-randomization of the masks of one bus (`+delta`, `-delta` on two members, no
/// reveal offsets) is just another honest choice of masks: accepted natively and in circuit.
/// This is also the control for the forgeries above: proofs made through the hook verify when
/// the binding and the per-bus sums hold.
#[test]
fn zk_npo_zero_sum_rerandomization_accepted() {
    let fx = multibus_d1();
    let seed = 51;
    let [a, b] = two_members(&fx.setup(seed).cpd, "PermuteBus");
    let mut seen = None;
    let mut hook = |plan: &mut LookupMaskPlan<Challenge>| {
        plan.committed[a.0][a.1] += delta();
        plan.committed[b.0][b.1] -= delta();
        seen = Some(plan.clone());
    };
    let p = fx.prove(seed, Some(&mut hook));
    let plan = seen.expect("the hook runs under ZK");
    assert!(
        plan.reveal_offsets
            .iter()
            .flatten()
            .all(|o| *o == Challenge::ZERO)
    );
    for (bus, sum) in per_bus_sums(&p) {
        assert_eq!(sum, Challenge::ZERO, "bus {bus} does not balance");
    }
    fx.verify_native(&p)
        .expect("re-randomized masks verify natively");
    fx.verify_in_circuit(&p)
        .expect("re-randomized masks verify in circuit");
}

/// Committed masks that do not sum to zero on a bus: the binding holds (the revealed value is
/// `mask + true sum`) but the bus no longer balances, so the per-bus check rejects. In a debug
/// build the core prover refuses to finish such a proof (see the `should_panic` twin).
fn non_zero_sum_masks() -> (NpoFixture<F, 1>, ZkProved) {
    let fx = multibus_d1();
    let seed = 61;
    let [a, _] = two_members(&fx.setup(seed).cpd, "PermuteBus");
    let mut hook = |plan: &mut LookupMaskPlan<Challenge>| {
        plan.committed[a.0][a.1] += delta();
    };
    let p = fx.prove(seed, Some(&mut hook));
    (fx, p)
}

#[cfg(not(debug_assertions))]
#[test]
fn zk_npo_non_zero_sum_masks_rejected() {
    let (fx, p) = non_zero_sum_masks();
    assert_eq!(per_bus_sums(&p)["PermuteBus"], delta());
    assert_native_rejects(&fx, &p, "GlobalCumulativeMismatch(None): PermuteBus");
    assert_in_circuit_rejects(&fx, &p, Check::PerBus);
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "do not balance")]
fn zk_npo_non_zero_sum_masks_refused_by_debug_prover() {
    let _ = non_zero_sum_masks();
}

/// Cross-bus mask shift: `+delta` on a PermuteBus mask and `-delta` on a RangeBus mask. The
/// bindings hold and the sum over all global lookups is still zero, so only a verifier that
/// checks every bus separately rejects it (natively and in circuit).
fn cross_bus_mask_shift() -> (NpoFixture<F, 1>, ZkProved) {
    let fx = multibus_d1();
    let seed = 71;
    let cpd = fx.setup(seed).cpd;
    let [a, _] = two_members(&cpd, "PermuteBus");
    let [b, _] = two_members(&cpd, "RangeBus");
    let mut hook = |plan: &mut LookupMaskPlan<Challenge>| {
        plan.committed[a.0][a.1] += delta();
        plan.committed[b.0][b.1] -= delta();
    };
    let p = fx.prove(seed, Some(&mut hook));
    (fx, p)
}

#[cfg(not(debug_assertions))]
#[test]
fn zk_npo_cross_bus_mask_shift_rejected() {
    let (fx, p) = cross_bus_mask_shift();
    let sums = per_bus_sums(&p);
    assert_eq!(sums["PermuteBus"], delta());
    assert_eq!(sums["RangeBus"], -delta());
    assert_eq!(
        sums.values().copied().sum::<Challenge>(),
        Challenge::ZERO,
        "the total over all buses still vanishes"
    );
    assert_native_rejects(&fx, &p, "GlobalCumulativeMismatch");
    assert_in_circuit_rejects(&fx, &p, Check::PerBus);
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "do not balance")]
fn zk_npo_cross_bus_mask_shift_refused_by_debug_prover() {
    let _ = cross_bus_mask_shift();
}
