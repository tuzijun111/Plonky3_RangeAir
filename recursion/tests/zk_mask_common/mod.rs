//! Shared fixtures for the zero-knowledge LogUp mask tests: a ZK test config, masked proofs of
//! circuits that use the NPO tables zkSketch depends on (Permute/PermuteBus, Range/RangeBus,
//! Expose with public values, Poseidon1), and their in-circuit verifier.
//!
//! Used by `zk_logup_mask_npo.rs` and `zk_mask_air_contract.rs`.

#![allow(dead_code)]

use std::collections::BTreeMap;

use p3_air::BaseAir;
use p3_batch_stark::ProverData;
use p3_circuit::ops::poseidon1_perm::KoalaBearD4Width16 as P1KoalaBearD4Width16;
use p3_circuit::ops::{
    Poseidon1Config, generate_expose_trace, generate_permute_trace, generate_poseidon1_trace,
    generate_poseidon2_trace, generate_range_trace, generate_recompose_trace,
};
use p3_circuit::{Circuit, CircuitBuilder, ExprId, NonPrimitiveOpId, Traces};
use p3_circuit_prover::batch_stark_prover::{
    BatchStarkProof, NUM_PRIMITIVE_TABLES, expose_air_builders, expose_preprocessor,
    expose_table_provers, permute_air_builders, permute_preprocessor, permute_table_provers,
    poseidon1_air_builders, poseidon1_table_provers_d4, range_air_builders, range_preprocessor,
    range_table_provers,
};
use p3_circuit_prover::common::{
    CircuitTableAir, NpoAirBuilder, NpoPreprocessor, get_airs_and_degrees_with_prep,
};
use p3_circuit_prover::{
    BatchStarkProver, BatchStarkProverError, CircuitProverData, ConstraintProfile,
    DEFAULT_ZK_MIN_TRACE_HEIGHT, LookupMaskPlan, TablePacking, TableProver, poseidon1_preprocessor,
};
use p3_field::PrimeCharacteristicRing;
use p3_fri::{FriParameters, HidingFriPcs};
use p3_koala_bear::default_koalabear_poseidon1_16;
use p3_lookup::logup::LogUpGadget;
use p3_lookup::{Kind, lookup_mask_width};
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

/// Test-only ZK config. The hiding RNG is a seeded `SmallRng`: each proof uses its own seed, so
/// masks differ across proofs. Seeded RNGs must never be used for real proofs.
pub type MyPcsZk = HidingFriPcs<F, Dft, MyMmcs, ChallengeMmcs, SmallRng>;
pub type MyConfigZk = StarkConfig<MyPcsZk, Challenge, Challenger>;
pub type InnerFriZk = HidingFriProofTargets<
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
pub type ValInputProof =
    InputProofTargets<F, Challenge, RecValMmcs<F, DIGEST_ELEMS, MyHash, MyCompress>>;

/// Challenger permutation of the in-circuit verifier.
pub const POSEIDON2: Poseidon2Config = Poseidon2Config::KOALA_BEAR_D4_W16;
/// Poseidon1 table of the D=4 fixture.
pub const POSEIDON1: Poseidon1Config = Poseidon1Config::KOALA_BEAR_D4_W16;

pub fn zk_config(seed: u64) -> MyConfigZk {
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
pub fn fri_verifier_params() -> FriVerifierParams {
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

/// A circuit, its traces, and the NPO tables it needs, at trace degree `TD`.
pub struct NpoFixture<EF, const TD: usize> {
    pub name: &'static str,
    pub circuit: Circuit<EF>,
    pub traces: Traces<EF>,
    pub preprocessors: fn() -> Vec<Box<dyn NpoPreprocessor<F>>>,
    pub air_builders: fn() -> Vec<Box<dyn NpoAirBuilder<MyConfigZk, TD>>>,
    /// Table provers for the prover, the native verifier and the in-circuit verifier.
    pub table_provers: fn() -> Vec<Box<dyn TableProver<MyConfigZk>>>,
}

fn d1_preprocessors() -> Vec<Box<dyn NpoPreprocessor<F>>> {
    vec![
        permute_preprocessor::<F>(),
        range_preprocessor::<F>(),
        expose_preprocessor::<F>(),
    ]
}

fn d1_air_builders() -> Vec<Box<dyn NpoAirBuilder<MyConfigZk, 1>>> {
    let mut b = permute_air_builders::<MyConfigZk, 1>(1);
    b.extend(range_air_builders::<MyConfigZk, 1>(1));
    b.extend(expose_air_builders::<MyConfigZk, 1>());
    b
}

fn d1_table_provers() -> Vec<Box<dyn TableProver<MyConfigZk>>> {
    let mut p = permute_table_provers::<MyConfigZk, 1>(1);
    p.extend(range_table_provers::<MyConfigZk, 1>(1));
    p.extend(expose_table_provers::<MyConfigZk, 1>());
    p
}

/// D=1 circuit on three buses: a 4-element Permute (PermuteBus), byte range checks with
/// `add_range` (RangeBus), ALU work and an Expose of two values (public values); every table
/// also binds its values on WitnessChecks.
pub fn multibus_d1() -> NpoFixture<F, 1> {
    let mut b = CircuitBuilder::<F>::new();
    b.enable_permute(generate_permute_trace::<F, F>);
    b.enable_range(generate_range_trace::<F, F>);
    b.enable_expose(generate_expose_trace::<F, F>);
    let inputs: Vec<ExprId> = (0..4).map(|_| b.public_input()).collect();
    let outs = b.add_permute(&inputs, vec![2, 0, 3, 1]).unwrap();
    let s01 = b.add(outs[0], outs[1]);
    let s = b.add(s01, outs[2]);
    b.add_range(&inputs).unwrap();
    b.add_expose(&[s, outs[3]]).unwrap();
    let circuit = b.build().unwrap();
    let mut runner = circuit.runner();
    runner
        .set_public_inputs(&[10, 20, 30, 40].map(F::from_u64))
        .unwrap();
    let traces = runner.run().unwrap();
    NpoFixture {
        name: "D=1 Permute+Range+Expose",
        circuit,
        traces,
        preprocessors: d1_preprocessors,
        air_builders: d1_air_builders,
        table_provers: d1_table_provers,
    }
}

fn d4_preprocessors() -> Vec<Box<dyn NpoPreprocessor<F>>> {
    vec![
        poseidon1_preprocessor::<F>(),
        permute_preprocessor::<F>(),
        range_preprocessor::<F>(),
        expose_preprocessor::<F>(),
    ]
}

fn d4_air_builders() -> Vec<Box<dyn NpoAirBuilder<MyConfigZk, 4>>> {
    let mut b = poseidon1_air_builders::<MyConfigZk, 4>();
    b.extend(permute_air_builders::<MyConfigZk, 4>(1));
    b.extend(range_air_builders::<MyConfigZk, 4>(1));
    b.extend(expose_air_builders::<MyConfigZk, 4>());
    b
}

fn d4_table_provers() -> Vec<Box<dyn TableProver<MyConfigZk>>> {
    let mut p = poseidon1_table_provers_d4::<MyConfigZk>(POSEIDON1);
    p.extend(permute_table_provers::<MyConfigZk, 4>(1));
    p.extend(range_table_provers::<MyConfigZk, 4>(1));
    p.extend(expose_table_provers::<MyConfigZk, 4>());
    p
}

/// D=4 circuit (extension-field trace) with a Poseidon1 permutation, a Permute, a 22-bit
/// `add_range_bits` check (the byte decomposition zkSketch uses) and an Expose.
pub fn npo_d4() -> NpoFixture<Challenge, 4> {
    let mut b = CircuitBuilder::<Challenge>::new();
    b.enable_poseidon1_perm::<P1KoalaBearD4Width16, _>(
        generate_poseidon1_trace::<Challenge, P1KoalaBearD4Width16>,
        default_koalabear_poseidon1_16(),
    );
    b.enable_permute(generate_permute_trace::<F, Challenge>);
    b.enable_range(generate_range_trace::<F, Challenge>);
    b.enable_expose(generate_expose_trace::<F, Challenge>);
    let inputs: Vec<ExprId> = (0..4).map(|_| b.public_input()).collect();
    let (_, hash) = b
        .add_poseidon1_perm_for_challenger_chained(POSEIDON1, None, &inputs)
        .unwrap();
    let outs = b.add_permute(&inputs, vec![1, 3, 0, 2]).unwrap();
    // 0 <= inputs[0] - inputs[1] = 3_000_000 < 2^22.
    let d = b.sub(inputs[0], inputs[1]);
    b.add_range_bits::<F>(d, 22).unwrap();
    let t = b.add(hash[0], outs[0]);
    b.add_expose(&[t, hash[1]]).unwrap();
    let circuit = b.build().unwrap();
    let mut runner = circuit.runner();
    runner
        .set_public_inputs(&[3_000_050, 50, 7, 9].map(Challenge::from_u64))
        .unwrap();
    let traces = runner.run().unwrap();
    NpoFixture {
        name: "D=4 Poseidon1+Permute+RangeBits+Expose",
        circuit,
        traces,
        preprocessors: d4_preprocessors,
        air_builders: d4_air_builders,
        table_provers: d4_table_provers,
    }
}

/// Packing for every ZK proof here: tables padded to the default hiding budget.
pub fn zk_packing() -> TablePacking {
    TablePacking::new(1, 2).with_min_trace_height(DEFAULT_ZK_MIN_TRACE_HEIGHT)
}

/// AIRs, prover data and AIR widths of a fixture under the ZK config.
pub struct ZkSetup<const TD: usize> {
    pub airs: Vec<CircuitTableAir<MyConfigZk, TD>>,
    pub cpd: CircuitProverData<MyConfigZk>,
    pub air_widths: Vec<usize>,
}

/// A masked ZK proof with what is needed to verify it.
pub struct ZkProved {
    pub proof: BatchStarkProof<MyConfigZk>,
    pub cpd: CircuitProverData<MyConfigZk>,
    pub air_widths: Vec<usize>,
}

/// Test-only mask hook (see `BatchStarkProver::prove_all_tables_with_mask_hook`).
pub type Hook<'a> = &'a mut dyn FnMut(&mut LookupMaskPlan<Challenge>);

impl<EF, const TD: usize> NpoFixture<EF, TD>
where
    EF: p3_field::Field
        + p3_field::ExtensionField<F>
        + p3_circuit_prover::field_params::ExtractBinomialW<F>,
{
    pub fn setup(&self, seed: u64) -> ZkSetup<TD> {
        let config = zk_config(seed);
        let (airs_degrees, pc, npc) = get_airs_and_degrees_with_prep::<MyConfigZk, EF, TD>(
            &self.circuit,
            &zk_packing(),
            &(self.preprocessors)(),
            &(self.air_builders)(),
            ConstraintProfile::Standard,
        )
        .unwrap();
        let (airs, degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();
        let air_widths = airs.iter().map(BaseAir::<F>::width).collect();
        let ext_degrees: Vec<usize> = degrees.iter().map(|&d| d + config.is_zk()).collect();
        let prover_data = ProverData::from_airs_and_degrees(&config, &airs, &ext_degrees);
        ZkSetup {
            airs,
            cpd: CircuitProverData::new(prover_data, pc, npc),
            air_widths,
        }
    }

    /// A prover (also the native verifier) for this fixture with hiding RNG seed `seed`.
    pub fn prover(&self, seed: u64) -> BatchStarkProver<MyConfigZk> {
        let mut prover = BatchStarkProver::new(zk_config(seed)).with_table_packing(zk_packing());
        for p in (self.table_provers)() {
            prover.register_table_prover(p);
        }
        prover
    }

    /// Prove the fixture with ZK (seed `seed`), optionally through the test-only mask hook.
    pub fn prove(&self, seed: u64, hook: Option<Hook<'_>>) -> ZkProved {
        let ZkSetup {
            cpd, air_widths, ..
        } = self.setup(seed);
        let prover = self.prover(seed);
        let proof = hook
            .map_or_else(
                || prover.prove_all_tables(&self.traces, &cpd),
                |h| prover.prove_all_tables_with_mask_hook(&self.traces, &cpd, |p| h(p)),
            )
            .expect("proving succeeds (forgeries are made by the hook, not refused by the prover)");
        ZkProved {
            proof,
            cpd,
            air_widths,
        }
    }

    /// Native verification with a fresh prover (another RNG seed; it is not used to verify).
    pub fn verify_native(&self, p: &ZkProved) -> Result<(), BatchStarkProverError> {
        self.prover(0xdead).verify_all_tables(&p.proof)
    }

    /// Build and run the in-circuit verifier of `p`; `Ok` iff all its checks pass.
    pub fn verify_in_circuit(&self, p: &ZkProved) -> Result<Traces<Challenge>, String> {
        let built =
            build_verifier::<TD>(p, &(self.table_provers)()).map_err(|e| format!("{e:?}"))?;
        run_verifier(&built, &p.proof)
    }
}

/// Global lookups of a proof's tables: `(table, global index, bus)`, in the order of the masks
/// (`LookupMaskPlan` and `global_lookup_data` are indexed `[table][global index]`).
pub fn global_members(cpd: &CircuitProverData<MyConfigZk>) -> Vec<(usize, usize, String)> {
    cpd.common_data()
        .lookups
        .iter()
        .enumerate()
        .flat_map(|(i, lookups)| {
            lookups
                .iter()
                .filter_map(|l| match &l.kind {
                    Kind::Global(name) => Some(name.clone()),
                    _ => None,
                })
                .enumerate()
                .map(move |(g, name)| (i, g, name))
        })
        .collect()
}

/// Number of global lookups per bus.
pub fn bus_sizes(cpd: &CircuitProverData<MyConfigZk>) -> BTreeMap<String, usize> {
    let mut m = BTreeMap::new();
    for (_, _, bus) in global_members(cpd) {
        *m.entry(bus).or_insert(0) += 1;
    }
    m
}

/// Two members `(table, global index)` of `bus`, preferring two different tables.
pub fn two_members(cpd: &CircuitProverData<MyConfigZk>, bus: &str) -> [(usize, usize); 2] {
    let m: Vec<(usize, usize)> = global_members(cpd)
        .into_iter()
        .filter(|(_, _, b)| b == bus)
        .map(|(i, g, _)| (i, g))
        .collect();
    assert!(m.len() >= 2, "bus {bus} needs two members, has {m:?}");
    let second = m[1..].iter().find(|x| x.0 != m[0].0).unwrap_or(&m[1]);
    [m[0], *second]
}

/// Revealed (masked) sums in `[table][global index]` order, flattened.
pub fn revealed_sums(p: &ZkProved) -> Vec<Challenge> {
    p.proof
        .proof
        .global_lookup_data
        .iter()
        .flatten()
        .map(|ld| ld.cumulative_sum)
        .collect()
}

/// Revealed (masked) sums per bus.
pub fn per_bus_sums(p: &ZkProved) -> BTreeMap<String, Challenge> {
    let mut sums = BTreeMap::new();
    for ld in p.proof.proof.global_lookup_data.iter().flatten() {
        *sums.entry(ld.name.clone()).or_insert(Challenge::ZERO) += ld.cumulative_sum;
    }
    sums
}

pub fn opened_main_widths(p: &ZkProved) -> Vec<usize> {
    p.proof
        .proof
        .opened_values
        .instances
        .iter()
        .map(|o| o.base_opened_values.trace_local.len())
        .collect()
}

/// Expected committed main widths: AIR width + `D` per global lookup.
pub fn expected_masked_widths(p: &ZkProved) -> Vec<usize> {
    p.air_widths
        .iter()
        .zip(p.cpd.common_data().lookups.iter())
        .map(|(w, l)| w + lookup_mask_width(l, D, true))
        .collect()
}

/// AIR public values per table: none for the primitive tables, the declared values for NPO
/// tables (only Expose has any).
pub fn table_public_values(proof: &BatchStarkProof<MyConfigZk>) -> Vec<Vec<F>> {
    let mut pis = vec![vec![]; NUM_PRIMITIVE_TABLES];
    pis.extend(proof.non_primitives.iter().map(|e| e.public_values.clone()));
    pis
}

/// An in-circuit verifier built for one proof, with the values that feed it.
pub struct BuiltVerifier {
    pub circuit: Circuit<Challenge>,
    pub public_inputs: Vec<Challenge>,
    pub private_inputs: Vec<Challenge>,
    pub mmcs_op_ids: Vec<NonPrimitiveOpId>,
}

pub fn verifier_circuit_builder() -> CircuitBuilder<Challenge> {
    let mut cb = CircuitBuilder::new();
    cb.enable_poseidon2_perm::<KoalaBearD4Width16, _>(
        generate_poseidon2_trace::<Challenge, KoalaBearD4Width16>,
        default_koalabear_poseidon2_16(),
    );
    cb.enable_recompose::<F>(generate_recompose_trace::<F, Challenge>);
    cb
}

/// Build the in-circuit verifier of a masked ZK proof whose trace field has degree `TD`.
pub fn build_verifier<const TD: usize>(
    p: &ZkProved,
    table_provers: &[Box<dyn TableProver<MyConfigZk>>],
) -> Result<BuiltVerifier, VerificationError> {
    let common = p.cpd.common_data();
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
        TD,
    >(
        &zk_config(999),
        &mut cb,
        &p.proof,
        &fri_verifier_params(),
        common,
        &LogUpGadget::new(),
        POSEIDON2,
        table_provers,
    )?;
    let circuit = cb.build().expect("verifier circuit builds");
    let (public_inputs, private_inputs) =
        vi.pack_values(&table_public_values(&p.proof), &p.proof.proof, common);
    Ok(BuiltVerifier {
        circuit,
        public_inputs,
        private_inputs,
        mmcs_op_ids,
    })
}

/// Run a built verifier circuit on the proof it was built for; `Ok` iff all its checks pass.
pub fn run_verifier(
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
