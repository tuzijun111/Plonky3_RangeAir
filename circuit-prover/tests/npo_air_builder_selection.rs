//! Which non-primitive AIR each registered `NpoAirBuilder` builds in
//! `get_airs_and_degrees_with_prep`.
//!
//! The plugin preprocessors return their rows in a `HashMap` keyed by `NpoTypeId`, whose
//! iteration order depends on a per-map random seed. Each builder used to take the first op type
//! it accepted in that order, so a builder accepting more than one of the circuit's op types
//! (e.g. `Poseidon2AirBuilder<4>` with both a D=4 and a D=1 Poseidon2 table in one circuit)
//! built a hash-order-dependent AIR, and with it a hash-order-dependent verifying key. Such a
//! builder is now refused, and a builder accepting exactly one op type always builds that one.

use std::any::Any;

use p3_baby_bear::BabyBear;
use p3_circuit::ops::{NonPrimitivePreprocessedMap, NpoTypeId};
use p3_circuit::{CircuitBuilder, CircuitError};
use p3_circuit_prover::air::ConstAir;
use p3_circuit_prover::common::{
    CircuitTableAir, NpoAirBuilder, NpoPreprocessor, get_airs_and_degrees_with_prep,
};
use p3_circuit_prover::config::BabyBearConfig;
use p3_circuit_prover::{ConstraintProfile, TablePacking};
use p3_field::PrimeCharacteristicRing;
use p3_field::extension::BinomialExtensionField;

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;
const D: usize = 4;

/// Op types the mock preprocessor reports, with the (distinct) degree the mock builders return
/// for each so the test can read back which op type an AIR was built from.
const OP_TYPES: [(&str, usize); 4] = [
    ("mock/alpha", 11),
    ("mock/bravo", 12),
    ("mock/charlie", 13),
    ("mock/delta", 14),
];

fn degree_of(op_type: &str) -> usize {
    OP_TYPES
        .iter()
        .find(|(name, _)| *name == op_type)
        .map(|&(_, deg)| deg)
        .expect("known mock op type")
}

/// Reports one row of preprocessed data for every entry of [`OP_TYPES`].
struct MockPreprocessor;

impl NpoPreprocessor<F> for MockPreprocessor {
    fn preprocess(
        &self,
        _circuit: &dyn Any,
        _preprocessed: &mut dyn Any,
    ) -> Result<NonPrimitivePreprocessedMap<F>, CircuitError> {
        Ok(OP_TYPES
            .iter()
            .map(|&(name, deg)| (NpoTypeId::new(name), vec![F::from_usize(deg)]))
            .collect())
    }
}

/// Accepts every op type whose name starts with `prefix`.
struct PrefixBuilder(&'static str);

impl NpoAirBuilder<BabyBearConfig, D> for PrefixBuilder {
    fn try_build(
        &self,
        op_type: &NpoTypeId,
        prep_base: &[F],
        _min_height: usize,
        _lanes: usize,
        _constraint_profile: ConstraintProfile,
    ) -> Option<(CircuitTableAir<BabyBearConfig, D>, usize)> {
        if !op_type.as_str().starts_with(self.0) {
            return None;
        }
        let deg = degree_of(op_type.as_str());
        assert_eq!(
            prep_base,
            &[F::from_usize(deg)],
            "builder gets its own op type's rows"
        );
        Some((CircuitTableAir::Const(ConstAir::new(1)), deg))
    }
}

fn npo_degrees(
    builders: &[Box<dyn NpoAirBuilder<BabyBearConfig, D>>],
) -> Result<Vec<usize>, CircuitError> {
    let mut builder = CircuitBuilder::<EF>::new();
    let x = builder.public_input();
    let _ = builder.mul(x, x);
    let circuit = builder.build().expect("circuit builds");

    let preprocessors: Vec<Box<dyn NpoPreprocessor<F>>> = vec![Box::new(MockPreprocessor)];
    let (airs_degrees, _, _) = get_airs_and_degrees_with_prep::<BabyBearConfig, EF, D>(
        &circuit,
        &TablePacking::default(),
        &preprocessors,
        builders,
        ConstraintProfile::Standard,
    )?;
    // The three primitive tables (Const, Public, Alu) come first.
    Ok(airs_degrees
        .into_iter()
        .skip(3)
        .map(|(_, deg)| deg)
        .collect())
}

/// A builder accepting several of the circuit's op types is refused instead of building
/// whichever the hash map yields first. Before the fix this returned `Ok` with one AIR whose op
/// type depended on the map's iteration order.
#[test]
fn a_builder_accepting_several_op_types_is_refused() {
    let builders: Vec<Box<dyn NpoAirBuilder<BabyBearConfig, D>>> =
        vec![Box::new(PrefixBuilder("mock/"))];
    for _ in 0..16 {
        let result = npo_degrees(&builders);
        assert!(
            matches!(result, Err(CircuitError::InvalidPreprocessing { .. })),
            "an ambiguous NPO AIR builder must be refused, got {result:?}"
        );
    }
}

/// Builders that each accept exactly one op type build it, in builder order, however the
/// preprocessed map happens to iterate (fresh map per call).
#[test]
fn each_builder_builds_its_own_op_type_in_builder_order() {
    let builders: Vec<Box<dyn NpoAirBuilder<BabyBearConfig, D>>> = vec![
        Box::new(PrefixBuilder("mock/charlie")),
        Box::new(PrefixBuilder("mock/alpha")),
        Box::new(PrefixBuilder("mock/nothing")),
        Box::new(PrefixBuilder("mock/delta")),
    ];
    let expected = vec![13, 11, 14];
    for _ in 0..16 {
        assert_eq!(
            npo_degrees(&builders).expect("unambiguous builders"),
            expected
        );
    }
}
