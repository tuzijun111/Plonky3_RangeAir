//! Permute non-primitive operation: proves multiset equality between an input and output
//! witness vector via a dedicated `PermuteBus` LogUp bus.
//!
//! The op carries a `permutation_witness: Vec<usize>` in [`NonPrimitiveOpParams::Permute`].
//! The executor uses it at trace-generation time to populate output witness values from
//! the inputs. The AIR proves only `multiset(inputs) == multiset(outputs)` — it does NOT
//! prove a specific permutation. Downstream primitives (LexSort, etc.) layer sortedness
//! or positional constraints on top of multiset equality to upgrade soundness.

use alloc::boxed::Box;
use alloc::format;
use alloc::string::ToString;
use alloc::vec::Vec;
use core::any::Any;
use core::fmt::Debug;
use core::marker::PhantomData;

use p3_field::{BasedVectorSpace, ExtensionField, Field, PrimeField64};

use crate::CircuitError;
use crate::builder::{
    CircuitBuilderError, NonPrimitiveOpParams, NpoCircuitPlugin, NpoLoweringContext,
};
use crate::ops::{ExecutionContext, NonPrimitiveExecutor, NpoTypeId, Op, PreprocessedWriter};
use crate::tables::{NonPrimitiveTrace, TraceGeneratorFn};
use crate::types::{ExprId, WitnessId};

// ============================================================================
// Execution State
// ============================================================================

/// Per-op data captured during execution.
#[derive(Debug, Clone)]
pub struct PermuteCircuitRow<F> {
    pub input_wids: Vec<WitnessId>,
    pub output_wids: Vec<WitnessId>,
    pub input_values: Vec<F>,
    pub output_values: Vec<F>,
}

/// Execution state collecting rows across all permute calls.
#[derive(Debug, Default)]
pub struct PermuteExecutionState<F> {
    pub rows: Vec<PermuteCircuitRow<F>>,
}

// ============================================================================
// Executor
// ============================================================================

/// Executor for a single permute op.
///
/// Per-op data: `permutation_witness` — the destination output index for each input position.
/// More precisely, `output_values[i] = input_values[permutation_witness[i]]`.
pub struct PermuteExecutor<F> {
    op_type: NpoTypeId,
    permutation_witness: Vec<usize>,
    _phantom: PhantomData<F>,
}

impl<F> Clone for PermuteExecutor<F> {
    fn clone(&self) -> Self {
        Self {
            op_type: self.op_type.clone(),
            permutation_witness: self.permutation_witness.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<F> Debug for PermuteExecutor<F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PermuteExecutor")
            .field("op_type", &self.op_type)
            .field("permutation_witness", &self.permutation_witness)
            .finish()
    }
}

impl<F> PermuteExecutor<F> {
    pub fn new(permutation_witness: Vec<usize>) -> Self {
        Self {
            op_type: NpoTypeId::permute(),
            permutation_witness,
            _phantom: PhantomData,
        }
    }
}

impl<F: Field + Send + Sync + 'static> NonPrimitiveExecutor<F> for PermuteExecutor<F> {
    fn execute(
        &self,
        inputs: &[Vec<WitnessId>],
        outputs: &[Vec<WitnessId>],
        ctx: &mut ExecutionContext<'_, F>,
    ) -> Result<(), CircuitError> {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(CircuitError::NonPrimitiveOpLayoutMismatch {
                op: self.op_type.clone(),
                expected: "1 input group and 1 output group".to_string(),
                got: inputs.len() + outputs.len(),
            });
        }
        let input_wids = &inputs[0];
        let output_wids = &outputs[0];
        let n = input_wids.len();
        if output_wids.len() != n {
            return Err(CircuitError::NonPrimitiveOpLayoutMismatch {
                op: self.op_type.clone(),
                expected: format!("output vector of length {n}"),
                got: output_wids.len(),
            });
        }
        if self.permutation_witness.len() != n {
            return Err(CircuitError::NonPrimitiveOpLayoutMismatch {
                op: self.op_type.clone(),
                expected: format!("permutation_witness of length {n}"),
                got: self.permutation_witness.len(),
            });
        }

        // Read input values.
        let mut input_values = Vec::with_capacity(n);
        for &wid in input_wids {
            input_values.push(ctx.get_witness(wid)?);
        }

        // Compute and write output values via the permutation.
        let mut output_values = Vec::with_capacity(n);
        for (i, &src) in self.permutation_witness.iter().enumerate() {
            if src >= n {
                return Err(CircuitError::NonPrimitiveOpLayoutMismatch {
                    op: self.op_type.clone(),
                    expected: format!("permutation_witness[{i}] in [0, {n})"),
                    got: src,
                });
            }
            let v = input_values[src];
            output_values.push(v);
            ctx.set_witness(output_wids[i], v)?;
        }

        // Record the row for trace generation.
        let state = ctx.get_op_state_mut::<PermuteExecutionState<F>>(&self.op_type);
        state.rows.push(PermuteCircuitRow {
            input_wids: input_wids.clone(),
            output_wids: output_wids.clone(),
            input_values,
            output_values,
        });

        Ok(())
    }

    fn op_type(&self) -> &NpoTypeId {
        &self.op_type
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    /// Preprocessed layout: 5 columns per (input, output) pair (one pair per AIR row).
    ///
    /// Per pair:
    /// - col 0: `input_idx`  — input witness index (also bumps `ext_reads[input_wid]`)
    /// - col 1: `input_mult` — constant `F::ONE` (each input is read once per row)
    /// - col 2: `output_idx` — output witness index (creator)
    /// - col 3: `output_mult` — placeholder `F::ONE`, patched by `PermutePreprocessor` to `ext_reads[output_wid]`
    /// - col 4: `is_real` — constant `F::ONE` for real rows, `F::ZERO` for padding (used as the
    ///   `PermuteBus` multiplier so padding rows contribute zero to the multiset balance)
    fn preprocess(
        &self,
        inputs: &[Vec<WitnessId>],
        outputs: &[Vec<WitnessId>],
        preprocessed: &mut dyn PreprocessedWriter<F>,
    ) -> Result<(), CircuitError> {
        let input_wids = &inputs[0];
        let output_wids = &outputs[0];
        for (&in_wid, &out_wid) in input_wids.iter().zip(output_wids.iter()) {
            // col 0: input_idx (+ bumps ext_reads)
            preprocessed.register_non_primitive_witness_reads(&self.op_type, &[in_wid])?;
            // col 1: input_mult = 1
            preprocessed.register_non_primitive_preprocessed_no_read(&self.op_type, &[F::ONE]);
            // col 2: output_idx
            preprocessed.register_non_primitive_output_index(&self.op_type, &[out_wid]);
            // col 3: output_mult placeholder (patched by PermutePreprocessor)
            preprocessed.register_non_primitive_preprocessed_no_read(&self.op_type, &[F::ONE]);
            // col 4: is_real = 1
            preprocessed.register_non_primitive_preprocessed_no_read(&self.op_type, &[F::ONE]);
        }
        Ok(())
    }

    fn num_exposed_outputs(&self) -> Option<usize> {
        Some(1)
    }

    fn boxed(&self) -> Box<dyn NonPrimitiveExecutor<F>> {
        Box::new(self.clone())
    }
}

// ============================================================================
// Circuit Plugin
// ============================================================================

/// Circuit-layer plugin for permute non-primitive operations.
///
/// Holds the trace generator function — typically `generate_permute_trace::<BF, EF>`
/// where `BF` is the base field used by the table prover and `EF` is the circuit's
/// field. The base-field-only smoke-test path uses `BF == EF`, matching the
/// original design before D > 1 support was added.
pub struct PermuteCircuitPlugin<F> {
    trace_gen: TraceGeneratorFn<F>,
    _phantom: PhantomData<F>,
}

impl<F> PermuteCircuitPlugin<F> {
    /// Build a plugin that, at trace-generation time, invokes `trace_gen` to
    /// convert the executor's per-row extension-field values into the base-field
    /// coefficient layout the `PermuteAir` consumes.
    pub fn new(trace_gen: TraceGeneratorFn<F>) -> Self {
        Self {
            trace_gen,
            _phantom: PhantomData,
        }
    }
}

impl<F> Debug for PermuteCircuitPlugin<F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("PermuteCircuitPlugin")
    }
}

// SAFETY: function pointers are `Send + Sync`, and `PhantomData<F>` is `Send + Sync`
// regardless of `F` — we store no `F` values.
unsafe impl<F> Send for PermuteCircuitPlugin<F> {}
unsafe impl<F> Sync for PermuteCircuitPlugin<F> {}

impl<F> NpoCircuitPlugin<F> for PermuteCircuitPlugin<F>
where
    F: Field + Send + Sync + 'static,
{
    fn type_id(&self) -> NpoTypeId {
        NpoTypeId::permute()
    }

    fn lower(
        &self,
        data: &crate::builder::NonPrimitiveOperationData<F>,
        output_exprs: &[(u32, ExprId)],
        ctx: &mut NpoLoweringContext<'_, F>,
    ) -> Result<Op<F>, CircuitBuilderError> {
        // Ensure every output expr has an allocated witness id.
        for (_idx, expr_id) in output_exprs {
            ctx.ensure_witness_id(*expr_id);
        }

        // Pull the permutation witness from per-op params.
        let permutation_witness: Vec<usize> = data
            .params
            .as_ref()
            .and_then(NonPrimitiveOpParams::as_permute)
            .ok_or_else(|| CircuitBuilderError::InvalidNonPrimitiveOpConfiguration {
                op: data.op_type.clone(),
            })?
            .to_vec();

        // Validate arity: one input group, matching outputs and perm length.
        if data.input_exprs.len() != 1 {
            return Err(CircuitBuilderError::NonPrimitiveOpArity {
                op: "Permute",
                expected: "1 input group".to_string(),
                got: data.input_exprs.len(),
            });
        }
        let input_wids: Vec<WitnessId> = data.input_exprs[0]
            .iter()
            .enumerate()
            .map(|(i, &expr)| ctx.resolve_witness_id(expr, &format!("Permute input {i}")))
            .collect::<Result<_, _>>()?;
        let n = input_wids.len();

        if output_exprs.len() != n {
            return Err(CircuitBuilderError::NonPrimitiveOpArity {
                op: "Permute",
                expected: format!("{n} outputs (one per input)"),
                got: output_exprs.len(),
            });
        }
        if permutation_witness.len() != n {
            return Err(CircuitBuilderError::NonPrimitiveOpArity {
                op: "Permute",
                expected: format!("permutation_witness of length {n}"),
                got: permutation_witness.len(),
            });
        }

        let output_wids: Vec<WitnessId> = output_exprs
            .iter()
            .enumerate()
            .map(|(i, &(_, expr))| ctx.resolve_witness_id(expr, &format!("Permute output {i}")))
            .collect::<Result<_, _>>()?;

        Ok(Op::NonPrimitiveOpWithExecutor {
            inputs: alloc::vec![input_wids],
            outputs: alloc::vec![output_wids],
            executor: Box::new(PermuteExecutor::<F>::new(permutation_witness)),
            op_id: data.op_id,
        })
    }

    fn trace_generator(&self) -> TraceGeneratorFn<F> {
        self.trace_gen
    }

    fn config(&self) -> crate::ops::NpoConfig {
        crate::ops::NpoConfig::new(())
    }
}

// ============================================================================
// Trace
// ============================================================================

/// Trace for permute operations.
#[derive(Debug, Clone)]
pub struct PermuteTrace<F> {
    pub operations: Vec<PermuteCircuitRow<F>>,
}

impl<F> PermuteTrace<F> {
    /// Total number of (input, output) pairs across all collected ops.
    pub fn total_rows(&self) -> usize {
        self.operations.iter().map(|r| r.input_wids.len()).sum()
    }
}

impl<TraceF: Clone + Send + Sync + Debug + 'static, CF> NonPrimitiveTrace<CF>
    for PermuteTrace<TraceF>
{
    fn op_type(&self) -> NpoTypeId {
        NpoTypeId::permute()
    }

    fn rows(&self) -> usize {
        self.total_rows()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn boxed_clone(&self) -> Box<dyn NonPrimitiveTrace<CF>> {
        Box::new(self.clone())
    }
}

// ============================================================================
// Trace Generation
// ============================================================================

/// Generate the permute trace from execution state.
///
/// `EF` is the circuit's field (the type the executor recorded values under);
/// `BF` is the base field the table prover consumes. Each extension-field
/// value is unpacked into its `D` base-field coefficients, so the resulting
/// `PermuteCircuitRow<BF>::input_values` has length `n_pairs · D` — `D`
/// consecutive base elements per logical permute slot, in coefficient order.
/// The `PermuteAir::trace_to_matrix` writer reads it back with the same stride.
///
/// For base-field circuits where `EF == BF` (e.g. the D=1 smoke test) the
/// conversion is a no-op identity: `as_basis_coefficients_slice` returns a
/// length-1 slice containing the same scalar.
pub fn generate_permute_trace<BF, EF>(
    op_states: &crate::ops::OpStateMap,
) -> Result<Option<Box<dyn NonPrimitiveTrace<EF>>>, CircuitError>
where
    BF: PrimeField64,
    EF: ExtensionField<BF>,
{
    let Some(state) = op_states
        .get(&NpoTypeId::permute())
        .and_then(|s| s.downcast_ref::<PermuteExecutionState<EF>>())
    else {
        return Ok(None);
    };

    if state.rows.is_empty() {
        return Ok(None);
    }

    let operations: Vec<PermuteCircuitRow<BF>> = state
        .rows
        .iter()
        .map(|row| {
            // Each EF value is unpacked into its `D` base-field coefficients.
            // For the D=1 case `as_basis_coefficients_slice` returns `[v]`, so
            // this preserves the original layout.
            let input_values: Vec<BF> = row
                .input_values
                .iter()
                .flat_map(|ef| {
                    <EF as BasedVectorSpace<BF>>::as_basis_coefficients_slice(ef)
                        .iter()
                        .copied()
                })
                .collect();
            let output_values: Vec<BF> = row
                .output_values
                .iter()
                .flat_map(|ef| {
                    <EF as BasedVectorSpace<BF>>::as_basis_coefficients_slice(ef)
                        .iter()
                        .copied()
                })
                .collect();
            PermuteCircuitRow {
                input_wids: row.input_wids.clone(),
                output_wids: row.output_wids.clone(),
                input_values,
                output_values,
            }
        })
        .collect();

    Ok(Some(Box::new(PermuteTrace { operations })))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_field::PrimeCharacteristicRing;
    use p3_test_utils::baby_bear_params::BabyBear;

    use super::*;
    use crate::builder::CircuitBuilder;

    type F = BabyBear;

    /// CircuitRunner-only smoke test of the executor + plugin (no STARK proof).
    ///
    /// Builds a circuit with 4 public inputs `[10, 20, 30, 40]`, applies
    /// `permutation_witness = [2, 0, 3, 1]` via `add_permute`, and checks the resulting
    /// output witness values are `[30, 10, 40, 20]`.
    #[test]
    fn permute_executor_round_trip() {
        let mut builder = CircuitBuilder::<F>::new();
        builder.enable_permute(generate_permute_trace::<F, F>);

        let inputs: Vec<ExprId> = (0..4).map(|_| builder.public_input()).collect();
        let outputs = builder
            .add_permute(&inputs, vec![2, 0, 3, 1])
            .expect("add_permute should accept a valid 4-element permutation");

        let circuit = builder.build().expect("build circuit");
        let out_wids: Vec<WitnessId> = outputs
            .iter()
            .map(|e| {
                *circuit
                    .expr_to_widx
                    .get(e)
                    .expect("output expr mapped to witness")
            })
            .collect();

        let mut runner = circuit.runner();
        runner
            .set_public_inputs(&[
                F::from_u64(10),
                F::from_u64(20),
                F::from_u64(30),
                F::from_u64(40),
            ])
            .expect("set public inputs");

        let traces = runner.run().expect("circuit runner executes");

        let expected = [
            F::from_u64(30),
            F::from_u64(10),
            F::from_u64(40),
            F::from_u64(20),
        ];
        for (i, (&wid, exp)) in out_wids.iter().zip(expected.iter()).enumerate() {
            let got = traces
                .witness_trace
                .get_value(wid)
                .expect("output witness assigned");
            assert_eq!(
                *got, *exp,
                "Permute output[{i}] mismatch: expected {exp:?}, got {got:?}"
            );
        }
    }

    /// Mismatched `permutation_witness` length must be rejected before execution.
    #[test]
    fn permute_rejects_length_mismatch() {
        let mut builder = CircuitBuilder::<F>::new();
        builder.enable_permute(generate_permute_trace::<F, F>);

        let inputs: Vec<ExprId> = (0..3).map(|_| builder.public_input()).collect();
        let err = builder
            .add_permute(&inputs, vec![0, 1])
            .expect_err("permutation_witness length 2 with 3 inputs should be rejected");

        match err {
            CircuitBuilderError::NonPrimitiveOpArity { op, .. } => assert_eq!(op, "Permute"),
            other => panic!("expected NonPrimitiveOpArity, got {other:?}"),
        }
    }
}
