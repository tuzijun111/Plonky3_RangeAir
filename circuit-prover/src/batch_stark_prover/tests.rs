use p3_baby_bear::BabyBear;
use p3_circuit::builder::CircuitBuilder;
use p3_circuit::ops::poseidon1_perm::{
    KoalaBearD1Width16 as P1KoalaBearD1Width16, Poseidon1PermCallBase,
};
use p3_circuit::ops::poseidon2_perm::{GoldilocksD2Width8, Poseidon2PermCallBase};
use p3_circuit::ops::{
    KoalaBearD1Width16, Poseidon1Config, Poseidon2Config, generate_poseidon1_trace,
    generate_poseidon2_trace, generate_recompose_trace,
};
use p3_field::PrimeCharacteristicRing;
use p3_field::extension::QuinticTrinomialExtensionField;
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
use p3_koala_bear::{KoalaBear, default_koalabear_poseidon1_16, default_koalabear_poseidon2_16};
use p3_symmetric::{CryptographicHasher, PaddingFreeSponge, Permutation};
use p3_test_utils::LiftPermToQuintic;

use super::*;
use crate::ConstraintProfile;
use crate::batch_stark_prover::packing::MAX_SANE_LANES;
use crate::batch_stark_prover::{
    BABY_BEAR_MODULUS, KOALA_BEAR_MODULUS, Poseidon1Preprocessor, Poseidon2Preprocessor,
    poseidon1_air_builders_d5, poseidon1_table_provers_d5, poseidon2_air_builders,
    poseidon2_air_builders_d5, poseidon2_table_provers_d5, recompose_air_builders,
};
use crate::common::{NpoPreprocessor, get_airs_and_degrees_with_prep};
use crate::config::{self, BabyBearConfig, GoldilocksConfig, KoalaBearConfig};

#[test]
fn test_babybear_batch_stark_base_field() {
    let mut builder = CircuitBuilder::<BabyBear>::new();

    // x + 5*2 - 3 + (-1) == expected
    let x = builder.public_input();
    let expected = builder.public_input();
    let c5 = builder.define_const(BabyBear::from_u64(5));
    let c2 = builder.define_const(BabyBear::from_u64(2));
    let c3 = builder.define_const(BabyBear::from_u64(3));
    let neg_one = builder.define_const(BabyBear::NEG_ONE);

    let mul_result = builder.mul(c5, c2); // 10
    let add_result = builder.add(x, mul_result); // x + 10
    let sub_result = builder.sub(add_result, c3); // x + 7
    let final_result = builder.add(sub_result, neg_one); // x + 6

    let diff = builder.sub(final_result, expected);
    builder.assert_zero(diff);

    let circuit = builder.build().unwrap();
    let cfg = config::baby_bear();
    let (airs_degrees, primitive_columns, non_primitive_columns) =
        get_airs_and_degrees_with_prep::<BabyBearConfig, _, 1>(
            &circuit,
            &TablePacking::default(),
            &[],
            &[],
            ConstraintProfile::Standard,
        )
        .unwrap();
    let (airs, log_degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();
    let prover_data = ProverData::from_airs_and_degrees(&cfg, &airs, &log_degrees);
    let circuit_prover_data =
        CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);

    let mut runner = circuit.runner();

    let x_val = BabyBear::from_u64(7);
    let expected_val = BabyBear::from_u64(13); // 7 + 10 - 3 - 1 = 13
    runner.set_public_inputs(&[x_val, expected_val]).unwrap();
    let traces = runner.run().unwrap();

    let prover = BatchStarkProver::new(cfg);

    let proof = prover
        .prove_all_tables(&traces, &circuit_prover_data)
        .unwrap();
    assert_eq!(proof.ext_degree, 1);
    assert!(proof.w_binomial.is_none());

    assert!(prover.verify_all_tables(&proof).is_ok());
}

#[test]
fn test_table_lookups() {
    let mut builder = CircuitBuilder::<BabyBear>::new();
    let cfg = config::baby_bear();

    // x + 5*2 - 3 + (-1) == expected
    let x = builder.public_input();
    let expected = builder.public_input();
    let c5 = builder.define_const(BabyBear::from_u64(5));
    let c2 = builder.define_const(BabyBear::from_u64(2));
    let c3 = builder.define_const(BabyBear::from_u64(3));
    let neg_one = builder.define_const(BabyBear::NEG_ONE);

    let mul_result = builder.mul(c5, c2); // 10
    let add_result = builder.add(x, mul_result); // x + 10
    let sub_result = builder.sub(add_result, c3); // x + 7
    let final_result = builder.add(sub_result, neg_one); // x + 6

    let diff = builder.sub(final_result, expected);
    builder.assert_zero(diff);

    let circuit = builder.build().unwrap();
    let default_packing = TablePacking::default();
    let (airs_degrees, primitive_columns, non_primitive_columns) =
        get_airs_and_degrees_with_prep::<BabyBearConfig, _, 1>(
            &circuit,
            &default_packing,
            &[],
            &[],
            ConstraintProfile::Standard,
        )
        .unwrap();
    let (airs, log_degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();

    let mut runner = circuit.runner();

    let x_val = BabyBear::from_u64(7);
    let expected_val = BabyBear::from_u64(13); // 7 + 10 - 3 - 1 = 13
    runner.set_public_inputs(&[x_val, expected_val]).unwrap();
    let traces = runner.run().unwrap();
    let prover_data = ProverData::from_airs_and_degrees(&cfg, &airs, &log_degrees);
    let circuit_prover_data =
        CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);

    let prover = BatchStarkProver::new(cfg);

    let proof = prover
        .prove_all_tables(&traces, &circuit_prover_data)
        .unwrap();
    assert_eq!(proof.ext_degree, 1);
    assert!(proof.w_binomial.is_none());

    assert!(prover.verify_all_tables(&proof).is_ok());

    // Check that the generated lookups are correct and consistent across tables.
    for air in airs.iter() {
        let lookups = crate::batch_stark_prover::lookups_for_circuit_table_air(air);

        match air {
            CircuitTableAir::Const(_) => {
                assert_eq!(lookups.len(), 1, "Const table should have one lookup");
            }
            CircuitTableAir::Public(_) => {
                assert_eq!(lookups.len(), 1, "Public table should have one lookup");
            }
            CircuitTableAir::Alu(_) => {
                // ALU table sends 4 lookups per lane + 2 extra for double-step Horner a1/c1
                let expected_num_lookups = default_packing.alu_lanes() * 4
                    + 2 * (default_packing.horner_packed_steps() - 1);
                assert_eq!(
                    lookups.len(),
                    expected_num_lookups,
                    "ALU table should have {} lookups, found {}",
                    expected_num_lookups,
                    lookups.len()
                );
            }
            CircuitTableAir::Dynamic(_dynamic_air) => {
                assert!(
                    lookups.is_empty(),
                    "There is no dynamic table in this test, so no lookups expected"
                );
            }
        }
    }
}

#[test]
fn test_extension_field_batch_stark() {
    const D: usize = 4;
    type Ext4 = BinomialExtensionField<BabyBear, D>;
    let cfg = config::baby_bear();

    let mut builder = CircuitBuilder::<Ext4>::new();
    let x = builder.public_input();
    let y = builder.public_input();
    let z = builder.public_input();
    let expected = builder.public_input();
    let xy = builder.mul(x, y);
    let res = builder.add(xy, z);
    let diff = builder.sub(res, expected);
    builder.assert_zero(diff);

    let circuit = builder.build().unwrap();
    let (airs_degrees, primitive_columns, non_primitive_columns) =
        get_airs_and_degrees_with_prep::<BabyBearConfig, _, D>(
            &circuit,
            &TablePacking::default(),
            &[],
            &[],
            ConstraintProfile::Standard,
        )
        .unwrap();
    let (airs, degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();

    let mut runner = circuit.runner();
    let xv = Ext4::from_basis_coefficients_slice(&[
        BabyBear::from_u64(2),
        BabyBear::from_u64(3),
        BabyBear::from_u64(5),
        BabyBear::from_u64(7),
    ])
    .unwrap();
    let yv = Ext4::from_basis_coefficients_slice(&[
        BabyBear::from_u64(11),
        BabyBear::from_u64(13),
        BabyBear::from_u64(17),
        BabyBear::from_u64(19),
    ])
    .unwrap();
    let zv = Ext4::from_basis_coefficients_slice(&[
        BabyBear::from_u64(23),
        BabyBear::from_u64(29),
        BabyBear::from_u64(31),
        BabyBear::from_u64(37),
    ])
    .unwrap();
    let expected_v = xv * yv + zv;
    runner.set_public_inputs(&[xv, yv, zv, expected_v]).unwrap();
    let traces = runner.run().unwrap();

    let prover_data = ProverData::from_airs_and_degrees(&cfg, &airs, &degrees);
    let circuit_prover_data =
        CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);
    let prover = BatchStarkProver::new(cfg);

    let proof = prover
        .prove_all_tables(&traces, &circuit_prover_data)
        .unwrap();
    assert_eq!(proof.ext_degree, 4);
    // Ensure W was captured
    let expected_w = <Ext4 as ExtractBinomialW<BabyBear>>::extract_w().unwrap();
    assert_eq!(proof.w_binomial, Some(expected_w));
    prover.verify_all_tables(&proof).unwrap();
}

#[test]
fn test_extension_field_table_lookups() {
    const D: usize = 4;
    type Ext4 = BinomialExtensionField<BabyBear, D>;
    let cfg = config::baby_bear();

    let mut builder = CircuitBuilder::<Ext4>::new();
    let x = builder.public_input();
    let y = builder.public_input();
    let z = builder.public_input();
    let expected = builder.public_input();
    let xy = builder.mul(x, y);
    let res = builder.add(xy, z);
    let diff = builder.sub(res, expected);
    builder.assert_zero(diff);

    let circuit = builder.build().unwrap();
    let default_packing = TablePacking::default();
    let mut air_builders_ext4 = poseidon2_air_builders::<BabyBearConfig, 4>();
    air_builders_ext4.extend(recompose_air_builders::<BabyBearConfig, 4>(1, false));
    let (airs_degrees, primitive_columns, non_primitive_columns) =
        get_airs_and_degrees_with_prep::<BabyBearConfig, _, D>(
            &circuit,
            &default_packing,
            &[],
            &air_builders_ext4,
            ConstraintProfile::Standard,
        )
        .unwrap();
    let (airs, log_degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();

    let mut runner = circuit.runner();

    let xv = Ext4::from_basis_coefficients_slice(&[
        BabyBear::from_u64(2),
        BabyBear::from_u64(3),
        BabyBear::from_u64(5),
        BabyBear::from_u64(7),
    ])
    .unwrap();
    let yv = Ext4::from_basis_coefficients_slice(&[
        BabyBear::from_u64(11),
        BabyBear::from_u64(13),
        BabyBear::from_u64(17),
        BabyBear::from_u64(19),
    ])
    .unwrap();
    let zv = Ext4::from_basis_coefficients_slice(&[
        BabyBear::from_u64(23),
        BabyBear::from_u64(29),
        BabyBear::from_u64(31),
        BabyBear::from_u64(37),
    ])
    .unwrap();
    let expected_v = xv * yv + zv;
    runner.set_public_inputs(&[xv, yv, zv, expected_v]).unwrap();
    let traces = runner.run().unwrap();

    let prover_data = ProverData::from_airs_and_degrees(&cfg, &airs, &log_degrees);
    let circuit_prover_data =
        CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);

    let prover = BatchStarkProver::new(cfg);

    let proof = prover
        .prove_all_tables(&traces, &circuit_prover_data)
        .unwrap();
    assert_eq!(proof.ext_degree, 4);
    // Ensure W was captured
    let expected_w = <Ext4 as ExtractBinomialW<BabyBear>>::extract_w().unwrap();
    assert_eq!(proof.w_binomial, Some(expected_w));

    assert!(prover.verify_all_tables(&proof).is_ok());

    // Check that the generated lookups are correct and consistent across tables.
    for air in airs.iter() {
        let lookups = crate::batch_stark_prover::lookups_for_circuit_table_air(air);

        match air {
            CircuitTableAir::Const(_) => {
                assert_eq!(lookups.len(), 1, "Const table should have one lookup");
            }
            CircuitTableAir::Public(_) => {
                assert_eq!(lookups.len(), 1, "Public table should have one lookup");
            }
            CircuitTableAir::Alu(_) => {
                // ALU table sends 4 lookups per lane + 2 extra for double-step Horner a1/c1
                let expected_num_lookups = default_packing.alu_lanes() * 4
                    + 2 * (default_packing.horner_packed_steps() - 1);
                assert_eq!(
                    lookups.len(),
                    expected_num_lookups,
                    "ALU table should have {} lookups, found {}",
                    expected_num_lookups,
                    lookups.len()
                );
            }
            CircuitTableAir::Dynamic(_dynamic_air) => {
                assert!(
                    lookups.is_empty(),
                    "There is no dynamic table in this test, so no lookups expected"
                );
            }
        }
    }
}

#[test]
fn test_koalabear_batch_stark_base_field() {
    let mut builder = CircuitBuilder::<KoalaBear>::new();
    let cfg = config::koala_bear();

    // a * b + 100 - (-1) == expected
    let a = builder.public_input();
    let b = builder.public_input();
    let expected = builder.public_input();
    let c = builder.define_const(KoalaBear::from_u64(100));
    let d = builder.define_const(KoalaBear::NEG_ONE);

    let ab = builder.mul(a, b);
    let add = builder.add(ab, c);
    let final_res = builder.sub(add, d);
    let diff = builder.sub(final_res, expected);
    builder.assert_zero(diff);

    let circuit = builder.build().unwrap();
    let (airs_degrees, primitive_columns, non_primitive_columns) =
        get_airs_and_degrees_with_prep::<KoalaBearConfig, _, 1>(
            &circuit,
            &TablePacking::default(),
            &[],
            &[],
            ConstraintProfile::Standard,
        )
        .unwrap();
    let (airs, degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();
    let mut runner = circuit.runner();

    let a_val = KoalaBear::from_u64(42);
    let b_val = KoalaBear::from_u64(13);
    let expected_val = KoalaBear::from_u64(647); // 42*13 + 100 - (-1)
    runner
        .set_public_inputs(&[a_val, b_val, expected_val])
        .unwrap();
    let traces = runner.run().unwrap();

    let prover_data = ProverData::from_airs_and_degrees(&cfg, &airs, &degrees);
    let circuit_prover_data =
        CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);
    let prover = BatchStarkProver::new(cfg);

    let proof = prover
        .prove_all_tables(&traces, &circuit_prover_data)
        .unwrap();
    assert_eq!(proof.ext_degree, 1);
    assert!(proof.w_binomial.is_none());
    prover.verify_all_tables(&proof).unwrap();
}

#[test]
fn test_koalabear_batch_stark_extension_field_d8() {
    const D: usize = 8;
    type KBExtField = BinomialExtensionField<KoalaBear, D>;
    let mut builder = CircuitBuilder::<KBExtField>::new();
    let cfg = config::koala_bear();

    // x * y * z == expected
    let x = builder.public_input();
    let y = builder.public_input();
    let expected = builder.public_input();
    let z = builder.define_const(
        KBExtField::from_basis_coefficients_slice(&[
            KoalaBear::from_u64(1),
            KoalaBear::NEG_ONE,
            KoalaBear::from_u64(2),
            KoalaBear::from_u64(3),
            KoalaBear::from_u64(4),
            KoalaBear::from_u64(5),
            KoalaBear::from_u64(6),
            KoalaBear::from_u64(7),
        ])
        .unwrap(),
    );

    let xy = builder.mul(x, y);
    let xyz = builder.mul(xy, z);
    let diff = builder.sub(xyz, expected);
    builder.assert_zero(diff);

    let circuit = builder.build().unwrap();
    let (airs_degrees, primitive_columns, non_primitive_columns) =
        get_airs_and_degrees_with_prep::<KoalaBearConfig, _, D>(
            &circuit,
            &TablePacking::default(),
            &[],
            &[],
            ConstraintProfile::Standard,
        )
        .unwrap();
    let (airs, degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();
    let mut runner = circuit.runner();

    let x_val = KBExtField::from_basis_coefficients_slice(&[
        KoalaBear::from_u64(4),
        KoalaBear::from_u64(6),
        KoalaBear::from_u64(8),
        KoalaBear::from_u64(10),
        KoalaBear::from_u64(12),
        KoalaBear::from_u64(14),
        KoalaBear::from_u64(16),
        KoalaBear::from_u64(18),
    ])
    .unwrap();
    let y_val = KBExtField::from_basis_coefficients_slice(&[
        KoalaBear::from_u64(12),
        KoalaBear::from_u64(14),
        KoalaBear::from_u64(16),
        KoalaBear::from_u64(18),
        KoalaBear::from_u64(20),
        KoalaBear::from_u64(22),
        KoalaBear::from_u64(24),
        KoalaBear::from_u64(26),
    ])
    .unwrap();
    let z_val = KBExtField::from_basis_coefficients_slice(&[
        KoalaBear::from_u64(1),
        KoalaBear::NEG_ONE,
        KoalaBear::from_u64(2),
        KoalaBear::from_u64(3),
        KoalaBear::from_u64(4),
        KoalaBear::from_u64(5),
        KoalaBear::from_u64(6),
        KoalaBear::from_u64(7),
    ])
    .unwrap();

    let expected_val = x_val * y_val * z_val;
    runner
        .set_public_inputs(&[x_val, y_val, expected_val])
        .unwrap();
    let traces = runner.run().unwrap();

    let prover_data = ProverData::from_airs_and_degrees(&cfg, &airs, &degrees);
    let circuit_prover_data =
        CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);
    let prover = BatchStarkProver::new(cfg);

    let proof = prover
        .prove_all_tables(&traces, &circuit_prover_data)
        .unwrap();
    assert_eq!(proof.ext_degree, 8);
    let expected_w = <KBExtField as ExtractBinomialW<KoalaBear>>::extract_w().unwrap();
    assert_eq!(proof.w_binomial, Some(expected_w));
    prover.verify_all_tables(&proof).unwrap();
}

#[test]
fn test_goldilocks_batch_stark_binomial_ext2() {
    const D: usize = 2;
    type Ext2 = BinomialExtensionField<Goldilocks, D>;
    let mut builder = CircuitBuilder::<Ext2>::new();
    let cfg = config::goldilocks();

    // x * y + z == expected
    let x = builder.public_input();
    let y = builder.public_input();
    let z = builder.public_input();
    let expected = builder.public_input();

    let xy = builder.mul(x, y);
    let res = builder.add(xy, z);
    let diff = builder.sub(res, expected);
    builder.assert_zero(diff);

    let circuit = builder.build().unwrap();
    let mut air_builders_ext2 = poseidon2_air_builders::<GoldilocksConfig, 2>();
    air_builders_ext2.extend(recompose_air_builders::<GoldilocksConfig, 2>(1, false));
    let (airs_degrees, primitive_columns, non_primitive_columns) =
        get_airs_and_degrees_with_prep::<GoldilocksConfig, _, D>(
            &circuit,
            &TablePacking::default(),
            &[],
            &air_builders_ext2,
            ConstraintProfile::Standard,
        )
        .unwrap();
    let (airs, degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();
    let mut runner = circuit.runner();

    let x_val =
        Ext2::from_basis_coefficients_slice(&[Goldilocks::from_u64(3), Goldilocks::NEG_ONE])
            .unwrap();
    let y_val =
        Ext2::from_basis_coefficients_slice(&[Goldilocks::from_u64(7), Goldilocks::from_u64(11)])
            .unwrap();
    let z_val =
        Ext2::from_basis_coefficients_slice(&[Goldilocks::from_u64(13), Goldilocks::from_u64(17)])
            .unwrap();
    let expected_val = x_val * y_val + z_val;

    runner
        .set_public_inputs(&[x_val, y_val, z_val, expected_val])
        .unwrap();
    let traces = runner.run().unwrap();

    let prover_data = ProverData::from_airs_and_degrees(&cfg, &airs, &degrees);
    let circuit_prover_data =
        CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);
    let prover = BatchStarkProver::new(cfg);

    let proof = prover
        .prove_all_tables(&traces, &circuit_prover_data)
        .unwrap();
    assert_eq!(proof.ext_degree, 2);
    let expected_w = <Ext2 as ExtractBinomialW<Goldilocks>>::extract_w().unwrap();
    assert_eq!(proof.w_binomial, Some(expected_w));
    prover.verify_all_tables(&proof).unwrap();
}

#[test]
fn test_goldilocks_poseidon2_circuit_build_and_run() {
    const D: usize = 2;
    type Ext2 = BinomialExtensionField<Goldilocks, D>;
    let mut rng = <rand::rngs::SmallRng as rand::SeedableRng>::seed_from_u64(0);
    let perm = Poseidon2Goldilocks::<8>::new_from_rng_128(&mut rng);
    let perm_for_hash = perm.clone();
    let mut builder = CircuitBuilder::<Ext2>::new();
    builder.enable_poseidon2_perm_width_8::<GoldilocksD2Width8, _>(
        generate_poseidon2_trace::<Ext2, GoldilocksD2Width8>,
        perm,
    );
    builder.enable_recompose::<Goldilocks>(generate_recompose_trace::<Goldilocks, Ext2>);
    let poseidon2_config = Poseidon2Config::GOLDILOCKS_D2_W8;
    let inputs = [builder.public_input(), builder.public_input()];
    let hash_outputs = builder
        .add_hash_slice(&poseidon2_config, &inputs, true)
        .unwrap();
    let expected0 = builder.public_input();
    let expected1 = builder.public_input();
    let sub0 = builder.sub(hash_outputs[0], expected0);
    builder.assert_zero(sub0);
    let sub1 = builder.sub(hash_outputs[1], expected1);
    builder.assert_zero(sub1);
    let circuit = builder.build().unwrap();
    let mut runner = circuit.runner();
    let in0 =
        Ext2::from_basis_coefficients_slice(&[Goldilocks::from_u64(1), Goldilocks::ZERO]).unwrap();
    let in1 =
        Ext2::from_basis_coefficients_slice(&[Goldilocks::from_u64(2), Goldilocks::ZERO]).unwrap();
    let hasher = PaddingFreeSponge::<Poseidon2Goldilocks<8>, 8, 4, 4>::new(perm_for_hash);
    let base_inputs = [
        Goldilocks::from_u64(1),
        Goldilocks::ZERO,
        Goldilocks::from_u64(2),
        Goldilocks::ZERO,
    ];
    let expected_hash = hasher.hash_iter(base_inputs);
    let out0 = Ext2::from_basis_coefficients_slice(&expected_hash[0..2]).unwrap();
    let out1 = Ext2::from_basis_coefficients_slice(&expected_hash[2..4]).unwrap();
    runner.set_public_inputs(&[in0, in1, out0, out1]).unwrap();
    let _traces = runner.run().unwrap();
}

#[test]
fn test_koalabear_modulus_constant() {
    // Verify KOALA_BEAR_MODULUS matches the actual KoalaBear field modulus.
    // The modulus p satisfies: from_u64(p) == 0 in the field.
    assert_eq!(
        KoalaBear::from_u64(KOALA_BEAR_MODULUS),
        KoalaBear::ZERO,
        "KOALA_BEAR_MODULUS (0x{:x}) does not match KoalaBear's actual modulus",
        KOALA_BEAR_MODULUS
    );

    // Verify the exact hex value (2130706433 = 0x7f000001).
    assert_eq!(KOALA_BEAR_MODULUS, 0x7f000001);
    assert_eq!(KOALA_BEAR_MODULUS, 2130706433);

    // Verify arithmetic at the modulus boundary with hardcoded expected values.
    // (p - 1) + 2 = 1 in the field
    let p_minus_1 = KoalaBear::from_u64(KOALA_BEAR_MODULUS - 1);
    assert_eq!(p_minus_1, KoalaBear::NEG_ONE);
    assert_eq!(p_minus_1 + KoalaBear::TWO, KoalaBear::ONE);

    // (p - 1) * (p - 1) = 1 in the field (since (-1) * (-1) = 1)
    assert_eq!(p_minus_1 * p_minus_1, KoalaBear::ONE);

    // Verify from_u64(p + 1) == 1
    assert_eq!(KoalaBear::from_u64(KOALA_BEAR_MODULUS + 1), KoalaBear::ONE);
}

#[test]
fn test_babybear_modulus_constant() {
    // Verify BABY_BEAR_MODULUS matches the actual BabyBear field modulus.
    assert_eq!(
        BabyBear::from_u64(BABY_BEAR_MODULUS),
        BabyBear::ZERO,
        "BABY_BEAR_MODULUS (0x{:x}) does not match BabyBear's actual modulus",
        BABY_BEAR_MODULUS
    );

    // Verify the exact hex value (2013265921 = 0x78000001).
    assert_eq!(BABY_BEAR_MODULUS, 0x78000001);
    assert_eq!(BABY_BEAR_MODULUS, 2013265921);

    // Verify arithmetic at the modulus boundary.
    let p_minus_1 = BabyBear::from_u64(BABY_BEAR_MODULUS - 1);
    assert_eq!(p_minus_1, BabyBear::NEG_ONE);
    assert_eq!(p_minus_1 + BabyBear::TWO, BabyBear::ONE);
    assert_eq!(BabyBear::from_u64(BABY_BEAR_MODULUS + 1), BabyBear::ONE);
}

#[test]
fn test_mul_only_circuit_padding() {
    // Circuit with only mul operations; ALU table still needs correct padding/lanes handling.
    let mut builder = CircuitBuilder::<BabyBear>::new();
    let cfg = config::baby_bear();

    let x = builder.public_input();
    let y = builder.public_input();

    // Only multiplication, no addition
    builder.mul(x, y);

    let circuit = builder.build().unwrap();
    let (airs_degrees, primitive_columns, non_primitive_columns) =
        get_airs_and_degrees_with_prep::<BabyBearConfig, _, 1>(
            &circuit,
            &TablePacking::default(),
            &[],
            &[],
            ConstraintProfile::Standard,
        )
        .unwrap();
    let (airs, degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();
    let mut runner = circuit.runner();

    let x_val = BabyBear::from_u64(7);
    let y_val = BabyBear::from_u64(11);
    runner.set_public_inputs(&[x_val, y_val]).unwrap();
    let traces = runner.run().unwrap();

    let prover_data = ProverData::from_airs_and_degrees(&cfg, &airs, &degrees);
    let circuit_prover_data =
        CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);

    let prover = BatchStarkProver::new(cfg);

    let proof = prover
        .prove_all_tables(&traces, &circuit_prover_data)
        .unwrap();
    prover.verify_all_tables(&proof).unwrap();
}

#[test]
fn test_add_only_circuit_padding() {
    // Circuit with only add operations; ALU table still needs correct padding/lanes handling.
    let mut builder = CircuitBuilder::<BabyBear>::new();
    let cfg = config::baby_bear();

    let x = builder.public_input();
    let y = builder.public_input();
    let expected = builder.public_input();

    // Only addition, no multiplication
    let sum = builder.add(x, y);
    let diff = builder.sub(sum, expected);
    builder.assert_zero(diff);

    let circuit = builder.build().unwrap();
    let (airs_degrees, primitive_columns, non_primitive_columns) =
        get_airs_and_degrees_with_prep::<BabyBearConfig, _, 1>(
            &circuit,
            &TablePacking::default(),
            &[],
            &[],
            ConstraintProfile::Standard,
        )
        .unwrap();
    let (airs, degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();
    let mut runner = circuit.runner();

    let x_val = BabyBear::from_u64(42);
    let y_val = BabyBear::from_u64(13);
    let expected_val = x_val + y_val;
    runner
        .set_public_inputs(&[x_val, y_val, expected_val])
        .unwrap();
    let traces = runner.run().unwrap();

    let prover_data = ProverData::from_airs_and_degrees(&cfg, &airs, &degrees);
    let circuit_prover_data =
        CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);

    let prover = BatchStarkProver::new(cfg);

    let proof = prover
        .prove_all_tables(&traces, &circuit_prover_data)
        .unwrap();
    prover.verify_all_tables(&proof).unwrap();
}

fn koala_ef5_lift(b: KoalaBear) -> QuinticTrinomialExtensionField<KoalaBear> {
    QuinticTrinomialExtensionField::<KoalaBear>::from_basis_coefficients_slice(&[
        b,
        KoalaBear::ZERO,
        KoalaBear::ZERO,
        KoalaBear::ZERO,
        KoalaBear::ZERO,
    ])
    .expect("basis slice")
}

#[test]
fn test_koalabear_quintic_trinomial_batch_stark_with_poseidon_d1() {
    const D: usize = 5;
    type EF5 = QuinticTrinomialExtensionField<KoalaBear>;

    // Must match KoalaBearD1Width16::round_constants() in poseidon2-circuit-air (not RNG-derived).
    let inner_perm = default_koalabear_poseidon2_16();
    let mut sponge0 = [KoalaBear::ZERO; 16];
    sponge0[0] = KoalaBear::from_u64(11);
    sponge0[1] = KoalaBear::from_u64(13);
    let sponge_out = inner_perm.permute(sponge0);
    let lift_perm = LiftPermToQuintic::new(inner_perm);

    let in0 = koala_ef5_lift(KoalaBear::from_u64(11));
    let in1 = koala_ef5_lift(KoalaBear::from_u64(13));
    let exp0 = koala_ef5_lift(sponge_out[0]);
    let exp1 = koala_ef5_lift(sponge_out[1]);

    let mut builder = CircuitBuilder::<EF5>::new();
    builder.enable_poseidon2_perm_base::<KoalaBearD1Width16, _>(
        generate_poseidon2_trace::<EF5, KoalaBearD1Width16>,
        lift_perm,
    );

    let in_a = builder.public_input();
    let in_b = builder.public_input();
    let mut perm_inputs: [Option<_>; 16] = [None; 16];
    perm_inputs[0] = Some(in_a);
    perm_inputs[1] = Some(in_b);
    let (_pid, hash_outputs) = builder
        .add_poseidon2_perm_base(&Poseidon2PermCallBase {
            config: Poseidon2Config::KOALA_BEAR_D1_W16,
            new_start: true,
            inputs: perm_inputs,
            // Only CTL-expose rate limbs that are wired into the rest of the circuit; unused
            // exposed outputs would leave WitnessChecks Receive contributions unmatched.
            out_ctl: [true; 8],
            return_all_outputs: false,
        })
        .unwrap();
    let e0 = builder.public_input();
    let e1 = builder.public_input();
    let h0_diff = builder.sub(hash_outputs[0].unwrap(), e0);
    let h1_diff = builder.sub(hash_outputs[1].unwrap(), e1);
    builder.assert_zero(h0_diff);
    builder.assert_zero(h1_diff);

    let circuit = builder.build().unwrap();
    let cfg = config::koala_bear();

    let npo_prep: Vec<Box<dyn NpoPreprocessor<KoalaBear>>> = vec![Box::new(Poseidon2Preprocessor)];
    let air_builders = poseidon2_air_builders_d5::<KoalaBearConfig>();
    let (airs_degrees, primitive_columns, non_primitive_columns) =
        get_airs_and_degrees_with_prep::<KoalaBearConfig, _, D>(
            &circuit,
            &TablePacking::default(),
            &npo_prep,
            &air_builders,
            ConstraintProfile::Standard,
        )
        .unwrap();
    let (airs, degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();
    let mut runner = circuit.runner();

    runner.set_public_inputs(&[in0, in1, exp0, exp1]).unwrap();
    let traces = runner.run().unwrap();

    let prover_data = ProverData::from_airs_and_degrees(&cfg, &airs, &degrees);
    let circuit_prover_data =
        CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);

    let mut prover = BatchStarkProver::new(cfg);
    for p in poseidon2_table_provers_d5(Poseidon2Config::KOALA_BEAR_D1_W16) {
        prover.register_table_prover(p);
    }

    let proof = prover
        .prove_all_tables(&traces, &circuit_prover_data)
        .unwrap();
    assert_eq!(proof.ext_degree, D);
    assert!(proof.w_binomial.is_none());
    assert!(proof.alu_quintic_trinomial);
    prover.verify_all_tables(&proof).unwrap();
}

/// Two D=1 Poseidon rows in an EF5 circuit: the second row uses `new_start=false` so the full
/// 16-wide state chains through the compact D=1 preprocessed layout (sponge selectors, not Merkle).
#[test]
fn test_koalabear_quintic_trinomial_batch_stark_poseidon_d1_sponge_chain() {
    const D: usize = 5;
    type EF5 = QuinticTrinomialExtensionField<KoalaBear>;

    let inner_perm = default_koalabear_poseidon2_16();
    let mut sponge0 = [KoalaBear::ZERO; 16];
    sponge0[0] = KoalaBear::from_u64(11);
    sponge0[1] = KoalaBear::from_u64(13);
    let sponge_out0 = inner_perm.permute(sponge0);
    let sponge_out1 = inner_perm.permute(sponge_out0);
    let lift_perm = LiftPermToQuintic::new(inner_perm);

    let in0 = koala_ef5_lift(KoalaBear::from_u64(11));
    let in1 = koala_ef5_lift(KoalaBear::from_u64(13));
    let exp0 = koala_ef5_lift(sponge_out1[0]);
    let exp1 = koala_ef5_lift(sponge_out1[1]);

    let mut builder = CircuitBuilder::<EF5>::new();
    builder.enable_poseidon2_perm_base::<KoalaBearD1Width16, _>(
        generate_poseidon2_trace::<EF5, KoalaBearD1Width16>,
        lift_perm,
    );

    let in_a = builder.public_input();
    let in_b = builder.public_input();
    let mut perm0_inputs: [Option<_>; 16] = [None; 16];
    perm0_inputs[0] = Some(in_a);
    perm0_inputs[1] = Some(in_b);
    let (_pid0, _hash0) = builder
        .add_poseidon2_perm_base(&Poseidon2PermCallBase {
            config: Poseidon2Config::KOALA_BEAR_D1_W16,
            new_start: true,
            inputs: perm0_inputs,
            out_ctl: [false; 8],
            return_all_outputs: false,
        })
        .unwrap();

    let perm1_inputs: [Option<_>; 16] = [None; 16];
    let (_pid1, hash1_outputs) = builder
        .add_poseidon2_perm_base(&Poseidon2PermCallBase {
            config: Poseidon2Config::KOALA_BEAR_D1_W16,
            new_start: false,
            inputs: perm1_inputs,
            out_ctl: [true; 8],
            return_all_outputs: false,
        })
        .unwrap();
    let e0 = builder.public_input();
    let e1 = builder.public_input();
    let h0_diff = builder.sub(hash1_outputs[0].unwrap(), e0);
    let h1_diff = builder.sub(hash1_outputs[1].unwrap(), e1);
    builder.assert_zero(h0_diff);
    builder.assert_zero(h1_diff);

    let circuit = builder.build().unwrap();
    let cfg = config::koala_bear();

    let npo_prep: Vec<Box<dyn NpoPreprocessor<KoalaBear>>> = vec![Box::new(Poseidon2Preprocessor)];
    let air_builders = poseidon2_air_builders_d5::<KoalaBearConfig>();
    let (airs_degrees, primitive_columns, non_primitive_columns) =
        get_airs_and_degrees_with_prep::<KoalaBearConfig, _, D>(
            &circuit,
            &TablePacking::default(),
            &npo_prep,
            &air_builders,
            ConstraintProfile::Standard,
        )
        .unwrap();
    let (airs, degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();
    let mut runner = circuit.runner();

    runner.set_public_inputs(&[in0, in1, exp0, exp1]).unwrap();
    let traces = runner.run().unwrap();

    let prover_data = ProverData::from_airs_and_degrees(&cfg, &airs, &degrees);
    let circuit_prover_data =
        CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);

    let mut prover = BatchStarkProver::new(cfg);
    for p in poseidon2_table_provers_d5(Poseidon2Config::KOALA_BEAR_D1_W16) {
        prover.register_table_prover(p);
    }

    let proof = prover
        .prove_all_tables(&traces, &circuit_prover_data)
        .unwrap();
    assert_eq!(proof.ext_degree, D);
    assert!(proof.w_binomial.is_none());
    assert!(proof.alu_quintic_trinomial);
    prover.verify_all_tables(&proof).unwrap();
}

#[test]
fn test_stark_serialization_round_trip() {
    let mut builder = CircuitBuilder::<BabyBear>::new();

    let x = builder.public_input();
    let expected = builder.public_input();
    let c5 = builder.define_const(BabyBear::from_u64(5));
    let c2 = builder.define_const(BabyBear::from_u64(2));
    let mul_result = builder.mul(c5, c2);
    let add_result = builder.add(x, mul_result);
    let diff = builder.sub(add_result, expected);
    builder.assert_zero(diff);

    let circuit = builder.build().unwrap();
    let cfg = config::baby_bear();
    let (airs_degrees, primitive_columns, non_primitive_columns) =
        get_airs_and_degrees_with_prep::<BabyBearConfig, _, 1>(
            &circuit,
            &TablePacking::default(),
            &[],
            &[],
            ConstraintProfile::Standard,
        )
        .unwrap();
    let (airs, log_degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();
    let prover_data = ProverData::from_airs_and_degrees(&cfg, &airs, &log_degrees);
    let circuit_prover_data =
        CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);

    let mut runner = circuit.runner();
    let x_val = BabyBear::from_u64(7);
    let expected_val = BabyBear::from_u64(17); // 7 + 5*2 = 17
    runner.set_public_inputs(&[x_val, expected_val]).unwrap();
    let traces = runner.run().unwrap();

    let prover = BatchStarkProver::new(cfg);
    let proof = prover
        .prove_all_tables(&traces, &circuit_prover_data)
        .unwrap();

    let original_preprocessed = proof
        .stark_common
        .preprocessed
        .as_ref()
        .expect("preprocessed binding must be present");
    let original_matrix_to_instance = original_preprocessed.matrix_to_instance.clone();
    let original_instances_len = original_preprocessed.instances.len();

    let bytes = postcard::to_allocvec(&proof).expect("serialize proof");
    let deserialized: BatchStarkProof<BabyBearConfig> =
        postcard::from_bytes(&bytes).expect("deserialize proof");

    let restored_preprocessed = deserialized
        .stark_common
        .preprocessed
        .as_ref()
        .expect("preprocessed binding must survive (de)serialization");
    assert_eq!(
        restored_preprocessed.matrix_to_instance,
        original_matrix_to_instance
    );
    assert_eq!(
        restored_preprocessed.instances.len(),
        original_instances_len
    );

    // Verification must succeed against the deserialized proof, relying only on the
    // proof's own `stark_common` for the preprocessed binding.
    prover
        .verify_all_tables(&deserialized)
        .expect("verification uses proof.stark_common");
}

// --- Proof-metadata validation after deserialization ---------------------------
//
// `#[derive(Deserialize)]` bypasses the constructors that enforce structural
// invariants (non-zero row counts, lane clamping, power-of-two minimum height,
// `horner_packed_steps >= 2`). These tests deserialize/construct the invalid
// states a malicious or corrupt serialized proof could carry and assert that
// `validate()` rejects them before verification.

/// Field-compatible mirror of `TablePacking` (same field order/types) used to
/// forge invalid serialized packings. `postcard` is non-self-describing, so a
/// structurally identical struct round-trips into the real `TablePacking`.
#[derive(serde::Serialize)]
struct PackingMirror {
    public_lanes: usize,
    alu_lanes: usize,
    npo_lanes: Vec<(p3_circuit::ops::NpoTypeId, usize)>,
    min_trace_height: usize,
    horner_packed_steps: usize,
}

impl PackingMirror {
    fn valid() -> Self {
        Self {
            public_lanes: 1,
            alu_lanes: 1,
            npo_lanes: Vec::new(),
            min_trace_height: 1,
            horner_packed_steps: 2,
        }
    }

    fn into_table_packing(self) -> TablePacking {
        let bytes = postcard::to_allocvec(&self).expect("serialize packing mirror");
        postcard::from_bytes(&bytes).expect("deserialize into TablePacking")
    }
}

#[test]
fn validate_rejects_zero_serialized_row_count() {
    // A `RowCounts` is a newtype over `[usize; N]`; derived `Deserialize` bypasses
    // `RowCounts::new`'s non-zero assertion.
    let bytes = postcard::to_allocvec(&[0usize, 1, 1]).expect("serialize raw row counts");
    let rows: RowCounts = postcard::from_bytes(&bytes).expect("deserialize RowCounts");
    assert_eq!(rows.validate(), Err(ProofMetadataError::ZeroRowCount));

    let ok = postcard::to_allocvec(&[1usize, 1, 1]).expect("serialize raw row counts");
    let rows: RowCounts = postcard::from_bytes(&ok).expect("deserialize RowCounts");
    assert_eq!(rows.validate(), Ok(()));
}

#[test]
fn validate_rejects_invalid_serialized_table_packing() {
    // Sanity: a valid mirror round-trips and validates.
    assert_eq!(
        PackingMirror::valid().into_table_packing().validate(),
        Ok(())
    );

    let zero_public = PackingMirror {
        public_lanes: 0,
        ..PackingMirror::valid()
    };
    assert_eq!(
        zero_public.into_table_packing().validate(),
        Err(ProofMetadataError::ZeroLanes("public_lanes"))
    );

    let zero_alu = PackingMirror {
        alu_lanes: 0,
        ..PackingMirror::valid()
    };
    assert_eq!(
        zero_alu.into_table_packing().validate(),
        Err(ProofMetadataError::ZeroLanes("alu_lanes"))
    );

    let op = p3_circuit::ops::NpoTypeId::new("test_op");
    let zero_npo = PackingMirror {
        npo_lanes: vec![(op.clone(), 0)],
        ..PackingMirror::valid()
    };
    assert_eq!(
        zero_npo.into_table_packing().validate(),
        Err(ProofMetadataError::ZeroNpoLanes(op))
    );

    let bad_height = PackingMirror {
        min_trace_height: 24, // not a power of two
        ..PackingMirror::valid()
    };
    assert_eq!(
        bad_height.into_table_packing().validate(),
        Err(ProofMetadataError::BadMinTraceHeight(24))
    );

    let zero_height = PackingMirror {
        min_trace_height: 0,
        ..PackingMirror::valid()
    };
    assert_eq!(
        zero_height.into_table_packing().validate(),
        Err(ProofMetadataError::BadMinTraceHeight(0))
    );

    let bad_horner = PackingMirror {
        horner_packed_steps: 1,
        ..PackingMirror::valid()
    };
    assert_eq!(
        bad_horner.into_table_packing().validate(),
        Err(ProofMetadataError::BadHornerPackedSteps(1))
    );
}

#[test]
fn validate_rejects_oversized_serialized_table_packing() {
    // Huge lane counts and horner_packed_steps drive downstream width arithmetic
    // (`lanes * lane_width`) and unbounded loops; a corrupt or malicious serialized
    // proof must be rejected here, not overflow/DoS the trace builder.
    let huge_public = PackingMirror {
        public_lanes: usize::MAX,
        ..PackingMirror::valid()
    };
    assert_eq!(
        huge_public.into_table_packing().validate(),
        Err(ProofMetadataError::LanesTooLarge {
            field: "public_lanes",
            got: usize::MAX,
            max: MAX_SANE_LANES,
        })
    );

    let huge_alu = PackingMirror {
        alu_lanes: usize::MAX,
        ..PackingMirror::valid()
    };
    assert_eq!(
        huge_alu.into_table_packing().validate(),
        Err(ProofMetadataError::LanesTooLarge {
            field: "alu_lanes",
            got: usize::MAX,
            max: MAX_SANE_LANES,
        })
    );

    let op = p3_circuit::ops::NpoTypeId::new("test_op");
    let huge_npo = PackingMirror {
        npo_lanes: vec![(op.clone(), usize::MAX)],
        ..PackingMirror::valid()
    };
    assert_eq!(
        huge_npo.into_table_packing().validate(),
        Err(ProofMetadataError::NpoLanesTooLarge {
            op_type: op,
            got: usize::MAX,
            max: MAX_SANE_LANES,
        })
    );

    let huge_horner = PackingMirror {
        horner_packed_steps: usize::MAX,
        ..PackingMirror::valid()
    };
    assert_eq!(
        huge_horner.into_table_packing().validate(),
        Err(ProofMetadataError::HornerPackedStepsTooLarge(
            usize::MAX,
            MAX_SANE_LANES,
        ))
    );
}

#[test]
fn validate_rejects_zero_lane_npo_entry() {
    // `NonPrimitiveTableEntry` has public fields, so deserialization can produce
    // a zero-lane entry directly.
    let op = p3_circuit::ops::NpoTypeId::new("test_op");
    let entry = NonPrimitiveTableEntry::<BabyBearConfig> {
        op_type: op.clone(),
        rows: 4,
        lanes: 0,
        public_values: Vec::new(),
        air_variant: AirVariant::Baseline,
    };
    assert_eq!(entry.validate(), Err(ProofMetadataError::ZeroNpoLanes(op)));
}

#[test]
fn verify_all_tables_rejects_tampered_serialized_row_counts() {
    // End-to-end: a real proof whose deserialized `rows` metadata was corrupted
    // to a zero count must be rejected before any AIR is reconstructed from it.
    let mut builder = CircuitBuilder::<BabyBear>::new();
    let x = builder.public_input();
    let expected = builder.public_input();
    let c5 = builder.define_const(BabyBear::from_u64(5));
    let c2 = builder.define_const(BabyBear::from_u64(2));
    let mul_result = builder.mul(c5, c2);
    let add_result = builder.add(x, mul_result);
    let diff = builder.sub(add_result, expected);
    builder.assert_zero(diff);

    let circuit = builder.build().unwrap();
    let cfg = config::baby_bear();
    let (airs_degrees, primitive_columns, non_primitive_columns) =
        get_airs_and_degrees_with_prep::<BabyBearConfig, _, 1>(
            &circuit,
            &TablePacking::default(),
            &[],
            &[],
            ConstraintProfile::Standard,
        )
        .unwrap();
    let (airs, log_degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();
    let prover_data = ProverData::from_airs_and_degrees(&cfg, &airs, &log_degrees);
    let circuit_prover_data =
        CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);

    let mut runner = circuit.runner();
    runner
        .set_public_inputs(&[BabyBear::from_u64(7), BabyBear::from_u64(17)])
        .unwrap();
    let traces = runner.run().unwrap();

    let prover = BatchStarkProver::new(cfg);
    let proof = prover
        .prove_all_tables(&traces, &circuit_prover_data)
        .unwrap();

    // Forge a deserialized proof with a zero primitive row count by serializing
    // raw counts and deserializing into `RowCounts` (a state `RowCounts::new`
    // would have rejected, but derived `Deserialize` accepts).
    let public_rows = proof.rows[PrimitiveTable::Public];
    let alu_rows = proof.rows[PrimitiveTable::Alu];
    let raw =
        postcard::to_allocvec(&[0usize, public_rows, alu_rows]).expect("serialize raw row counts");
    let tampered_rows: RowCounts =
        postcard::from_bytes(&raw).expect("deserialize tampered RowCounts");
    let tampered = BatchStarkProof {
        rows: tampered_rows,
        ..proof
    };

    let err = prover
        .verify_all_tables(&tampered)
        .expect_err("tampered row counts must be rejected before verification");
    assert!(
        matches!(
            err,
            BatchStarkProverError::InvalidMetadata(ProofMetadataError::ZeroRowCount)
        ),
        "unexpected error: {err:?}"
    );
}

/// Full prove/verify round-trip of a single D=1 Poseidon1 permutation in an EF5 circuit.
#[test]
fn test_koalabear_quintic_trinomial_batch_stark_with_poseidon1_d1() {
    const D: usize = 5;
    type EF5 = QuinticTrinomialExtensionField<KoalaBear>;

    // Must match `KoalaBearD1Width16::round_constants()` in poseidon1-circuit-air.
    let inner_perm = default_koalabear_poseidon1_16();
    let mut sponge0 = [KoalaBear::ZERO; 16];
    sponge0[0] = KoalaBear::from_u64(11);
    sponge0[1] = KoalaBear::from_u64(13);
    let sponge_out = inner_perm.permute(sponge0);
    let lift_perm = LiftPermToQuintic::new(inner_perm);

    let in0 = koala_ef5_lift(KoalaBear::from_u64(11));
    let in1 = koala_ef5_lift(KoalaBear::from_u64(13));
    let exp0 = koala_ef5_lift(sponge_out[0]);
    let exp1 = koala_ef5_lift(sponge_out[1]);

    let mut builder = CircuitBuilder::<EF5>::new();
    builder.enable_poseidon1_perm_base::<P1KoalaBearD1Width16, _>(
        generate_poseidon1_trace::<EF5, P1KoalaBearD1Width16>,
        lift_perm,
    );

    let in_a = builder.public_input();
    let in_b = builder.public_input();
    let mut perm_inputs: [Option<_>; 16] = [None; 16];
    perm_inputs[0] = Some(in_a);
    perm_inputs[1] = Some(in_b);
    let (_pid, hash_outputs) = builder
        .add_poseidon1_perm_base(&Poseidon1PermCallBase {
            config: Poseidon1Config::KOALA_BEAR_D1_W16,
            new_start: true,
            inputs: perm_inputs,
            out_ctl: [true; 8],
            return_all_outputs: false,
        })
        .unwrap();
    let e0 = builder.public_input();
    let e1 = builder.public_input();
    let h0_diff = builder.sub(hash_outputs[0].unwrap(), e0);
    let h1_diff = builder.sub(hash_outputs[1].unwrap(), e1);
    builder.assert_zero(h0_diff);
    builder.assert_zero(h1_diff);

    let circuit = builder.build().unwrap();
    let cfg = config::koala_bear();

    let npo_prep: Vec<Box<dyn NpoPreprocessor<KoalaBear>>> = vec![Box::new(Poseidon1Preprocessor)];
    let air_builders = poseidon1_air_builders_d5::<KoalaBearConfig>();
    let (airs_degrees, primitive_columns, non_primitive_columns) =
        get_airs_and_degrees_with_prep::<KoalaBearConfig, _, D>(
            &circuit,
            &TablePacking::default(),
            &npo_prep,
            &air_builders,
            ConstraintProfile::Standard,
        )
        .unwrap();
    let (airs, degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();
    let mut runner = circuit.runner();

    runner.set_public_inputs(&[in0, in1, exp0, exp1]).unwrap();
    let traces = runner.run().unwrap();

    let prover_data = ProverData::from_airs_and_degrees(&cfg, &airs, &degrees);
    let circuit_prover_data =
        CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);

    let mut prover = BatchStarkProver::new(cfg);
    for p in poseidon1_table_provers_d5(Poseidon1Config::KOALA_BEAR_D1_W16) {
        prover.register_table_prover(p);
    }

    let proof = prover
        .prove_all_tables(&traces, &circuit_prover_data)
        .unwrap();
    assert_eq!(proof.ext_degree, D);
    assert!(proof.w_binomial.is_none());
    assert!(proof.alu_quintic_trinomial);
    prover.verify_all_tables(&proof).unwrap();
}

// Regression tests for the constant-binding soundness bug (adapted from upstream
// 01743b9 / bdf7262 / 573f16e to this fork's non-generic `verify_all_tables`, plus a
// BabyBear D=4 case, the extension setting zkSketch uses).
//
// A compile-time constant (`define_const` / `alloc_const`) must be part of the circuit's
// committed preprocessing. At 19dcb37 `ConstAir`'s preprocessed columns carried only
// `[multiplicity, witness_idx]`, so a trace generated from a circuit whose constant had been
// swapped verified against the ORIGINAL circuit's preprocessed commitment.
mod forged_constant_value {
    extern crate std;

    use p3_circuit::ops::Op;
    use p3_field::extension::BinomialExtensionField;

    use super::*;

    /// Proves `traces` against `data` and verifies the result. Evaluates to `Err` when the
    /// proof is rejected, either by the prover's debug-only `check_constraints` panic or by
    /// `verify_all_tables`. Any other panic fails the test.
    macro_rules! prove_then_verify {
        ($prover:expr, $traces:expr, $data:expr) => {{
            let (prover, traces, data) = (&$prover, &$traces, &$data);
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let proof = prover
                    .prove_all_tables(traces, data)
                    .map_err(|e| alloc::format!("prove: {e:?}"))?;
                assert!(
                    proof.stark_common.preprocessed.as_ref().map(|p| &p.commitment)
                        == data.common_data().preprocessed.as_ref().map(|p| &p.commitment),
                    "the proof must carry the preprocessed commitment of the prover data it used"
                );
                prover
                    .verify_all_tables(&proof)
                    .map_err(|e| alloc::format!("verify: {e:?}"))
            }))
            .unwrap_or_else(|payload| {
                let msg = payload
                    .downcast_ref::<alloc::string::String>()
                    .map(alloc::string::String::as_str)
                    .or_else(|| payload.downcast_ref::<&str>().copied())
                    .unwrap_or("<non-string panic>");
                assert!(
                    msg.contains("constraints not satisfied on row"),
                    "expected the ConstAir constraint-violation panic, got a different panic: {msg}"
                );
                Err(alloc::string::String::from(
                    "prover panicked on the forged constant",
                ))
            })
        }};
    }

    fn swap_const<F: Field>(
        circuit: &p3_circuit::Circuit<F>,
        from: F,
        to: F,
    ) -> p3_circuit::Circuit<F> {
        let mut forged = circuit.clone();
        let mut changes = 0;
        for op in &mut forged.ops {
            if let Op::Const { val, .. } = op
                && *val == from
            {
                *val = to;
                changes += 1;
            }
        }
        assert_eq!(
            changes, 1,
            "exactly one Const op should carry the swapped value"
        );
        forged
    }

    #[test]
    fn verify_all_tables_rejects_a_forged_constant_value_d1() {
        let mut builder = CircuitBuilder::<KoalaBear>::new();
        let c = builder.define_const(KoalaBear::from_u32(42));
        let x = builder.public_input();
        builder.connect(c, x);
        let circuit = builder.build().unwrap();

        let mk = |circ: &p3_circuit::Circuit<KoalaBear>| {
            let cfg = config::koala_bear();
            let (ad, pc, npc) = get_airs_and_degrees_with_prep::<KoalaBearConfig, _, 1>(
                circ,
                &TablePacking::default(),
                &[],
                &[],
                ConstraintProfile::Standard,
            )
            .unwrap();
            let (airs, degs): (Vec<_>, Vec<usize>) = ad.into_iter().unzip();
            CircuitProverData::new(
                ProverData::from_airs_and_degrees(&cfg, &airs, &degs),
                pc,
                npc,
            )
        };
        let honest_data = mk(&circuit);
        let prover = BatchStarkProver::new(config::koala_bear());

        let mut r = circuit.runner();
        r.set_public_inputs(&[KoalaBear::from_u32(42)]).unwrap();
        let honest_traces = r.run().unwrap();
        prove_then_verify!(prover, honest_traces, honest_data)
            .expect("honest proof must verify");

        let forged = swap_const(&circuit, KoalaBear::from_u32(42), KoalaBear::from_u32(43));
        assert_ne!(
            circuit.generate_preprocessed_columns::<1>().unwrap(),
            forged.generate_preprocessed_columns::<1>().unwrap(),
            "circuits differing only in a constant's value must have different preprocessing"
        );
        let forged_data = mk(&forged);
        assert!(
            honest_data.common_data().preprocessed.as_ref().unwrap().commitment
                != forged_data.common_data().preprocessed.as_ref().unwrap().commitment,
            "circuits differing only in a constant's value must commit to different \
             preprocessed data"
        );

        let mut fr = forged.runner();
        fr.set_public_inputs(&[KoalaBear::from_u32(43)]).unwrap();
        let forged_traces = fr.run().unwrap();
        // The exploit: prove the forged trace against the ORIGINAL circuit's prover data, so
        // the proof carries the honest preprocessed commitment (VK).
        let res = prove_then_verify!(prover, forged_traces, honest_data);
        std::println!("forged D1 constant: {res:?}");
        assert!(
            res.is_err(),
            "a trace asserting the wrong value for a compile-time constant must be rejected"
        );
    }

    #[test]
    fn verify_all_tables_rejects_a_forged_constant_value_d4() {
        type EF = BinomialExtensionField<BabyBear, 4>;
        const D: usize = 4;
        let val = EF::from_basis_coefficients_fn(|i| BabyBear::from_u32(10 + i as u32));
        let forged_val = EF::from_basis_coefficients_fn(|i| BabyBear::from_u32(20 + i as u32));

        let mut builder = CircuitBuilder::<EF>::new();
        let c = builder.define_const(val);
        let x = builder.public_input();
        let y = builder.mul(c, x);
        let out = builder.public_input();
        builder.connect(y, out);
        let circuit = builder.build().unwrap();

        let mk = |circ: &p3_circuit::Circuit<EF>| {
            let cfg = config::baby_bear();
            let (ad, pc, npc) = get_airs_and_degrees_with_prep::<BabyBearConfig, _, D>(
                circ,
                &TablePacking::default(),
                &[],
                &[],
                ConstraintProfile::Standard,
            )
            .unwrap();
            let (airs, degs): (Vec<_>, Vec<usize>) = ad.into_iter().unzip();
            CircuitProverData::new(
                ProverData::from_airs_and_degrees(&cfg, &airs, &degs),
                pc,
                npc,
            )
        };
        let honest_data = mk(&circuit);
        let prover = BatchStarkProver::new(config::baby_bear());
        let xin = EF::from_basis_coefficients_fn(|i| BabyBear::from_u32(3 + i as u32));

        let mut r = circuit.runner();
        r.set_public_inputs(&[xin, val * xin]).unwrap();
        let honest_traces = r.run().unwrap();
        prove_then_verify!(prover, honest_traces, honest_data)
            .expect("honest D4 proof must verify");

        let forged = swap_const(&circuit, val, forged_val);
        let forged_data = mk(&forged);
        assert!(
            honest_data.common_data().preprocessed.as_ref().unwrap().commitment
                != forged_data.common_data().preprocessed.as_ref().unwrap().commitment,
            "D4: circuits differing only in a constant's value must commit to different \
             preprocessed data"
        );

        let mut fr = forged.runner();
        fr.set_public_inputs(&[xin, forged_val * xin]).unwrap();
        let forged_traces = fr.run().unwrap();
        let res = prove_then_verify!(prover, forged_traces, honest_data);
        std::println!("forged D4 constant: {res:?}");
        assert!(
            res.is_err(),
            "D4: a trace asserting the wrong value for a compile-time constant must be rejected"
        );
    }
}

// Regression test for the non-canonical full-width `decompose_to_bits` soundness gap
// (upstream 71449a3 / 9e682ef), in the BabyBear D=4 setting zkSketch uses.
//
// `decompose_to_bits::<BF>(x, BF::bits())` only constrained the bits to be boolean and to
// reconstruct to `x` modulo `p`. For BabyBear `p = 15 * 2^27 + 1 < 2^31`, so every `x <= 2^27 - 2`
// also decomposes as the 31-bit integer `x + p`, whose low 27 bits read `x + 1`. The hint is
// honest, so the test swaps the compiled circuit's decomposition hint for one that writes the
// `x + p` alias (a choice a prover assembling trace polynomials directly is free to make; the
// preprocessed columns are asserted unchanged), then proves the resulting traces against the
// HONEST circuit's prover data and verifies. The forged statement is "the low byte of 5 is 6".
mod non_canonical_bits {
    extern crate std;

    use alloc::boxed::Box;
    use alloc::vec::Vec;

    use p3_circuit::ops::{HintExecutor, Op};
    use p3_circuit::{CircuitError, WitnessId};
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{BasedVectorSpace, PrimeField64};

    use super::*;

    type EF = BinomialExtensionField<BabyBear, 4>;
    const D: usize = 4;

    /// Writes the bits of `x + p` (the non-canonical alias) instead of the bits of `x`.
    #[derive(Debug, Clone)]
    struct AliasedBits;

    impl HintExecutor<EF> for AliasedBits {
        fn execute(
            &self,
            inputs: &[WitnessId],
            outputs: &[WitnessId],
            witness: &mut [Option<EF>],
        ) -> Result<(), CircuitError> {
            let x = witness[inputs[0].0 as usize].expect("input set");
            let x0 = <EF as BasedVectorSpace<BabyBear>>::as_basis_coefficients_slice(&x)[0]
                .as_canonical_u64();
            let alias = x0 + BabyBear::ORDER_U64;
            assert!(alias < (1u64 << outputs.len()), "alias must fit the bit width");
            for (i, &out) in outputs.iter().enumerate() {
                witness[out.0 as usize] = Some(EF::from_u64((alias >> i) & 1));
            }
            Ok(())
        }

        fn boxed(&self) -> Box<dyn HintExecutor<EF>> {
            Box::new(self.clone())
        }
    }

    #[test]
    fn full_width_decompose_to_bits_rejects_the_x_plus_p_alias_d4() {
        let n_bits = <BabyBear as Field>::bits();
        let mut builder = CircuitBuilder::<EF>::new();
        let x = builder.public_input();
        let bits = builder.decompose_to_bits::<BabyBear>(x, n_bits).unwrap();
        let low = builder
            .reconstruct_index_from_bits::<BabyBear>(&bits[..8])
            .unwrap();
        let low_pub = builder.public_input();
        builder.connect(low, low_pub);
        let circuit = builder.build().unwrap();

        let cfg = config::baby_bear();
        let (ad, pc, npc) = get_airs_and_degrees_with_prep::<BabyBearConfig, _, D>(
            &circuit,
            &TablePacking::default(),
            &[],
            &[],
            ConstraintProfile::Standard,
        )
        .unwrap();
        let (airs, degs): (Vec<_>, Vec<usize>) = ad.into_iter().unzip();
        let honest_data =
            CircuitProverData::new(ProverData::from_airs_and_degrees(&cfg, &airs, &degs), pc, npc);
        let prover = BatchStarkProver::new(config::baby_bear());

        // Honest: low byte of 5 is 5.
        let mut r = circuit.runner();
        r.set_public_inputs(&[EF::from_u64(5), EF::from_u64(5)]).unwrap();
        let honest_traces = r.run().unwrap();
        let hp = prover.prove_all_tables(&honest_traces, &honest_data).unwrap();
        prover.verify_all_tables(&hp).expect("honest proof must verify");

        // Forged: swap the decomposition hint for the `x + p` alias.
        let mut forged = circuit.clone();
        let mut swapped = 0;
        for op in &mut forged.ops {
            if let Op::Hint {
                outputs, executor, ..
            } = op
                && outputs.len() == n_bits
            {
                *executor = Box::new(AliasedBits);
                swapped += 1;
            }
        }
        assert_eq!(swapped, 1, "exactly one 31-bit decomposition hint");
        assert_eq!(
            circuit.generate_preprocessed_columns::<D>().unwrap(),
            forged.generate_preprocessed_columns::<D>().unwrap(),
            "swapping a hint executor must not change the committed preprocessing"
        );

        let mut fr = forged.runner();
        fr.set_public_inputs(&[EF::from_u64(5), EF::from_u64(6)]).unwrap();
        let outcome: Result<(), alloc::string::String> = match fr.run() {
            Err(e) => Err(alloc::format!("witness generation rejected: {e:?}")),
            Ok(forged_traces) => {
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let fp = prover
                        .prove_all_tables(&forged_traces, &honest_data)
                        .map_err(|e| alloc::format!("prove: {e:?}"))?;
                    prover
                        .verify_all_tables(&fp)
                        .map_err(|e| alloc::format!("verify: {e:?}"))
                }))
                .unwrap_or_else(|_| Err("prover panicked".into()))
            }
        };
        std::println!("non-canonical alias outcome: {outcome:?}");
        assert!(
            outcome.is_err(),
            "a full-width decomposition must not accept the x + p alias (forged low byte 6 of 5)"
        );
    }
}
