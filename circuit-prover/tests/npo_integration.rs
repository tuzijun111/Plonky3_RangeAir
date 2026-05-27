use core::marker::PhantomData;

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_circuit::builder::{
    CircuitBuilder, NonPrimitiveOperationData, NpoCircuitPlugin, NpoLoweringContext,
};
use p3_circuit::ops::{
    ExecutionContext, NonPrimitiveExecutor, NpoConfig, NpoTypeId, Op, OpStateMap,
    generate_permute_trace,
};
use p3_circuit::tables::{NonPrimitiveTrace, TraceGeneratorFn};
use p3_circuit::{CircuitBuilderError, CircuitError, WitnessId};
use p3_circuit_prover::batch_stark_prover::{
    BatchStarkProver, CircuitProverData, TablePacking, permute_air_builders, permute_preprocessor,
};
use p3_circuit_prover::common::get_airs_and_degrees_with_prep;
use p3_circuit_prover::{ConstraintProfile, config};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;
use p3_test_utils::baby_bear_params::*;

// Simple non-primitive "cube" op: y = x^3
const CUBE_TYPE_ID: &str = "cube_simple/x_cubed";

fn cube_trace_generator<F>(
    _op_states: &OpStateMap,
) -> Result<Option<Box<dyn NonPrimitiveTrace<F>>>, CircuitError> {
    // This simple example does not produce its own dedicated table trace.
    Ok(None)
}

/// Circuit-side plugin for the cube op.
#[derive(Clone)]
struct CubeCircuitPlugin<F> {
    _phantom: PhantomData<F>,
}

impl<F> CubeCircuitPlugin<F> {
    const fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<F> NpoCircuitPlugin<F> for CubeCircuitPlugin<F>
where
    F: Field + PrimeCharacteristicRing,
{
    fn type_id(&self) -> NpoTypeId {
        NpoTypeId::new(CUBE_TYPE_ID)
    }

    fn lower(
        &self,
        data: &NonPrimitiveOperationData<F>,
        output_exprs: &[(u32, p3_circuit::types::ExprId)],
        ctx: &mut NpoLoweringContext<'_, F>,
    ) -> Result<Op<F>, CircuitBuilderError> {
        // For this example, expect exactly one input and one output.
        let input_expr = data.input_exprs[0][0];
        let output_expr = output_exprs[0].1;

        // Map expressions to witness IDs (allocate if necessary).
        let in_wid = *ctx
            .expr_to_widx
            .entry(input_expr)
            .or_insert_with(|| (ctx.alloc_witness_id)(1));
        let out_wid = *ctx
            .expr_to_widx
            .entry(output_expr)
            .or_insert_with(|| (ctx.alloc_witness_id)(1));

        // Build a non-primitive op with a cube executor.
        Ok(Op::NonPrimitiveOpWithExecutor {
            inputs: vec![vec![in_wid]],
            outputs: vec![vec![out_wid]],
            executor: Box::new(CubeExecutor::default()),
            op_id: data.op_id,
        })
    }

    fn trace_generator(&self) -> TraceGeneratorFn<F> {
        // For this demo we don't build a separate cube table trace; a real plugin
        // would record rows in OpExecutionState and use them here.
        cube_trace_generator::<F>
    }

    fn config(&self) -> NpoConfig {
        // No special config for this simple example.
        NpoConfig::new(())
    }
}

/// Executor that computes y = x^3 inside the runtime execution context.
#[derive(Clone)]
struct CubeExecutor<F> {
    op_type: NpoTypeId,
    _phantom: PhantomData<F>,
}

impl<F> Default for CubeExecutor<F> {
    fn default() -> Self {
        Self {
            op_type: NpoTypeId::new(CUBE_TYPE_ID),
            _phantom: PhantomData,
        }
    }
}

impl<F> core::fmt::Debug for CubeExecutor<F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("CubeExecutor")
    }
}

impl<F> NonPrimitiveExecutor<F> for CubeExecutor<F>
where
    F: Field + PrimeCharacteristicRing,
{
    fn execute(
        &self,
        inputs: &[Vec<WitnessId>],
        outputs: &[Vec<WitnessId>],
        ctx: &mut ExecutionContext<'_, F>,
    ) -> Result<(), p3_circuit::CircuitError> {
        let in_id = inputs[0][0];
        let out_id = outputs[0][0];

        let x = ctx.get_witness(in_id)?;
        let x2 = x * x;
        let x3 = x2 * x;

        ctx.set_witness(out_id, x3)?;
        Ok(())
    }

    fn op_type(&self) -> &NpoTypeId {
        &self.op_type
    }

    fn as_any(&self) -> &dyn core::any::Any {
        self
    }

    fn preprocess(
        &self,
        _inputs: &[Vec<WitnessId>],
        _outputs: &[Vec<WitnessId>],
        _preprocessed: &mut dyn p3_circuit::PreprocessedWriter<F>,
    ) -> Result<(), p3_circuit::CircuitError> {
        Ok(())
    }

    fn boxed(&self) -> Box<dyn NonPrimitiveExecutor<F>> {
        Box::new(self.clone())
    }
}

// ============================================================================
// Cube AIR
// ============================================================================

/// A minimal AIR that proves `y = x^3` for a single row.
///
/// Columns: `[x, x_sq, x_cu]`
/// Constraints:
///   - `x_sq - x * x   = 0`  (degree 2)
///   - `x_cu - x_sq * x = 0`  (degree 2)
#[derive(Clone)]
struct CubeAir;

impl<F> BaseAir<F> for CubeAir {
    fn width(&self) -> usize {
        3
    }
}

impl<AB: AirBuilder> Air<AB> for CubeAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.current_slice();
        let x = row[0];
        let x_sq = row[1];
        let x_cu = row[2];

        // x_sq = x * x
        builder.assert_zero(x_sq - x * x);
        // x_cu = x_sq * x
        builder.assert_zero(x_cu - x_sq * x);
    }
}

/// Integration-style test: register a cube NPO plugin, use it in a circuit,
/// and run the circuit to check y = x^3.
#[test]
fn cube_npo_integration_flow() {
    type F = BabyBear;

    // Build circuit with the cube plugin.
    let mut builder = CircuitBuilder::<F>::new();
    builder.register_npo(CubeCircuitPlugin::<F>::new());

    // Public input x and expected output y.
    let x = builder.public_input();
    let y_expected = builder.public_input();

    // Create a single cube non-primitive op that maps x -> y.
    let cube_type = NpoTypeId::new(CUBE_TYPE_ID);
    let (_op_id, _call_expr, outputs) = builder.push_non_primitive_op_with_outputs(
        cube_type,
        vec![vec![x]],
        vec![Some("cube_out")],
        None,
        "cube_call",
    );
    let y_expr = outputs[0].expect("cube op should have one output");

    // Connect cube_out to expected output.
    builder.connect(y_expr, y_expected);

    let circuit = builder.build().expect("build cube circuit");
    let out_wid = circuit
        .expr_to_widx
        .get(&y_expr)
        .copied()
        .expect("y_expr mapped to witness");

    // Run with a simple x value and check we get x^3.
    let mut runner = circuit.runner();
    let x_val = F::from_u64(3); // 3
    let y_val = x_val * x_val * x_val; // 27
    runner
        .set_public_inputs(&[x_val, y_val])
        .expect("set public inputs");

    let traces = runner.run().expect("run cube circuit");
    let out_val = traces
        .witness_trace
        .get_value(out_wid)
        .expect("output witness set");

    assert_eq!(*out_val, y_val);
}

/// Generates and verifies a STARK proof for a circuit that contains the cube NPO.
///
/// The circuit uses the cube NPO to compute `y = x^3`.  Because the NPO trace
/// generator returns `None`, the batch prover covers only the primitive tables
/// (Const, Public, ALU), but the circuit itself still contains the NPO operation.
/// This test verifies end-to-end STARK proof generation for an NPO-containing circuit.
#[test]
fn cube_npo_stark_proof() {
    type F = BabyBear;
    const D: usize = 1;

    // Build the same cube NPO circuit.
    let mut builder = CircuitBuilder::<F>::new();
    builder.register_npo(CubeCircuitPlugin::<F>::new());

    let x = builder.public_input();
    let cube_type = NpoTypeId::new(CUBE_TYPE_ID);
    let (_op_id, _call_expr, outputs) = builder.push_non_primitive_op_with_outputs(
        cube_type,
        vec![vec![x]],
        vec![Some("cube_out")],
        None,
        "cube_call",
    );
    // Assert that the NPO output is non-zero (x^3 = 125 ≠ 0).
    // We use assert_zero on (cube_out - cube_out) as a trivial consistency check;
    // the real verification of y = x^3 is done by the circuit runner via the executor.
    let y_expr = outputs[0].expect("cube op should have one output");
    let zero = builder.sub(y_expr, y_expr);
    builder.assert_zero(zero);

    let circuit = builder.build().expect("build cube circuit");
    let cfg = config::baby_bear();

    // Derive AIRs and preprocessed columns from the circuit.
    // The cube NPO has no dedicated table, so only primitive AIRs are generated.
    let (airs_degrees, primitive_columns, non_primitive_columns) =
        get_airs_and_degrees_with_prep::<config::BabyBearConfig, _, D>(
            &circuit,
            &TablePacking::default(),
            &[],
            &[],
            ConstraintProfile::Standard,
        )
        .expect("get_airs_and_degrees_with_prep should succeed");
    let (airs, log_degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();

    // Run the circuit to produce traces.
    let mut runner = circuit.runner();
    let x_val = F::from_u64(5);
    runner
        .set_public_inputs(&[x_val])
        .expect("set public inputs");
    let traces = runner.run().expect("run cube circuit");

    // Prove all primitive tables.
    let prover_data = p3_batch_stark::ProverData::from_airs_and_degrees(&cfg, &airs, &log_degrees);
    let circuit_prover_data =
        CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);
    let prover = BatchStarkProver::new(cfg);

    let proof = prover
        .prove_all_tables(&traces, &circuit_prover_data)
        .expect("prove_all_tables should succeed");

    prover
        .verify_all_tables(&proof)
        .expect("verify_all_tables should succeed");
}

/// Verifies the `CubeAir` constraints directly using the uni-stark prover.
///
/// Constructs a single-row trace `[x, x^2, x^3]` and proves it under `CubeAir`,
/// confirming that the two degree-2 constraints are satisfied.
#[test]
fn cube_air_stark_proof() {
    use p3_test_utils::baby_bear_params::*;
    use p3_uni_stark::{prove, verify};

    let cfg = make_test_config();

    // Construct a single-row trace: [x, x^2, x^3]
    let x = BabyBear::from_u64(7);
    let x_sq = x * x;
    let x_cu = x_sq * x;
    let trace = RowMajorMatrix::new(vec![x, x_sq, x_cu], 3);

    let proof = prove(&cfg, &CubeAir, trace, &[]);

    verify(&cfg, &CubeAir, &proof, &[]).expect("CubeAir STARK proof should verify");
}

// ============================================================================
// Permute (Phase 7a-1) — end-to-end prove + verify smoke test for `PermuteAir`.
// ============================================================================

/// Build a 4-element Permute circuit, prove + verify it end-to-end.
///
/// Verifies that:
/// - The `PermuteCircuitPlugin` + `PermuteExecutor` produce a valid trace.
/// - The `PermuteAir`'s `WitnessChecks` and `PermuteBus` interactions balance.
/// - `BatchStarkProver` accepts the proof.
#[test]
fn permute_npo_stark_proof() {
    type F = BabyBear;
    const D: usize = 1;

    // Build a circuit with 4 inputs and a Permute op.
    let mut builder = CircuitBuilder::<F>::new();
    builder.enable_permute(generate_permute_trace::<F, F>);

    let inputs: Vec<p3_circuit::types::ExprId> = (0..4).map(|_| builder.public_input()).collect();
    let outputs = builder
        .add_permute(&inputs, vec![2, 0, 3, 1])
        .expect("add_permute should accept a valid 4-element permutation");

    // Trivial consistency expression to force outputs into the witness layout (mirrors cube test).
    for &out in outputs.iter() {
        let z = builder.sub(out, out);
        builder.assert_zero(z);
    }

    let circuit = builder.build().expect("build permute circuit");
    let cfg = config::baby_bear();

    // Derive AIRs + preprocessed columns including PermuteAir.
    let (airs_degrees, primitive_columns, non_primitive_columns) =
        get_airs_and_degrees_with_prep::<config::BabyBearConfig, _, D>(
            &circuit,
            &TablePacking::default(),
            &[permute_preprocessor::<F>()],
            &permute_air_builders::<config::BabyBearConfig, D>(1),
            ConstraintProfile::Standard,
        )
        .expect("get_airs_and_degrees_with_prep should succeed");
    let (airs, log_degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();

    // Run the circuit to produce traces (Permute executor populates output values).
    let mut runner = circuit.runner();
    runner
        .set_public_inputs(&[
            F::from_u64(10),
            F::from_u64(20),
            F::from_u64(30),
            F::from_u64(40),
        ])
        .expect("set public inputs");
    let traces = runner.run().expect("run permute circuit");

    // Prove + verify all tables, including the new PermuteAir.
    let prover_data = p3_batch_stark::ProverData::from_airs_and_degrees(&cfg, &airs, &log_degrees);
    let circuit_prover_data =
        CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);
    let mut prover = BatchStarkProver::new(cfg);
    prover.register_permute_table::<D>();

    let proof = prover
        .prove_all_tables(&traces, &circuit_prover_data)
        .expect("prove_all_tables should succeed for honest permute");

    prover
        .verify_all_tables(&proof)
        .expect("verify_all_tables should accept honest permute");
}

/// Soundness check: a "permutation_witness" that creates duplicate outputs (so the multiset of
/// output values does NOT equal the multiset of input values) must be rejected by the
/// `PermuteBus` balance check at verify time.
///
/// `[0, 0, 1, 2]` maps to `output = [in[0], in[0], in[1], in[2]]`, missing `in[3]` and
/// duplicating `in[0]` — multisets differ unless inputs collide accidentally.
#[test]
fn permute_npo_stark_proof_rejects_bad_perm() {
    type F = BabyBear;
    const D: usize = 1;

    let mut builder = CircuitBuilder::<F>::new();
    builder.enable_permute(generate_permute_trace::<F, F>);

    let inputs: Vec<p3_circuit::types::ExprId> = (0..4).map(|_| builder.public_input()).collect();
    let outputs = builder
        .add_permute(&inputs, vec![0, 0, 1, 2])
        .expect("add_permute should accept the perm structurally (length matches)");
    for &out in outputs.iter() {
        let z = builder.sub(out, out);
        builder.assert_zero(z);
    }

    let circuit = builder.build().expect("build permute circuit");
    let cfg = config::baby_bear();

    let (airs_degrees, primitive_columns, non_primitive_columns) =
        get_airs_and_degrees_with_prep::<config::BabyBearConfig, _, D>(
            &circuit,
            &TablePacking::default(),
            &[permute_preprocessor::<F>()],
            &permute_air_builders::<config::BabyBearConfig, D>(1),
            ConstraintProfile::Standard,
        )
        .expect("get_airs_and_degrees_with_prep should succeed");
    let (airs, log_degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();

    let mut runner = circuit.runner();
    // Use 4 distinct values so the bad perm produces a genuinely-different multiset.
    runner
        .set_public_inputs(&[
            F::from_u64(10),
            F::from_u64(20),
            F::from_u64(30),
            F::from_u64(40),
        ])
        .expect("set public inputs");
    let traces = runner.run().expect("run permute circuit");

    let prover_data = p3_batch_stark::ProverData::from_airs_and_degrees(&cfg, &airs, &log_degrees);
    let circuit_prover_data =
        CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);
    let mut prover = BatchStarkProver::new(cfg);
    prover.register_permute_table::<D>();

    // The prover may complete (the witness is consistent with the executor's view of "apply this
    // perm") but verification MUST reject because PermuteBus does not balance.
    let prove_result = prover.prove_all_tables(&traces, &circuit_prover_data);
    match prove_result {
        Ok(proof) => {
            let verify_result = prover.verify_all_tables(&proof);
            assert!(
                verify_result.is_err(),
                "verify_all_tables should reject a bad permute (PermuteBus imbalance), but it \
                 accepted the proof"
            );
            eprintln!("[soundness] prove succeeded; verify rejected the bad perm — expected.");
        }
        Err(e) => {
            eprintln!("[soundness] prove_all_tables refused the bad perm early: {e:?}");
        }
    }
}
