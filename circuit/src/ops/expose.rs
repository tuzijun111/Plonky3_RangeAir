//! Expose non-primitive operation: reads K input witnesses and surfaces their
//! values as the table instance's `public_values`, so they propagate as
//! recursion `air_public_targets` when the proof is verified recursively.
//!
//! Unlike every other NPO, [`ExposeAir`](crate) declares `num_public_values()`
//! and the table prover sets `BatchTableInstance.public_values` to the K input
//! values — this is the only channel that carries an application value from a
//! verified child into the parent aggregation's public statement.
//!
//! The op has **no outputs**: each input wire is read on the shared
//! `WitnessChecks` bus (a reader, bumping `ext_reads` on its creator), and the
//! AIR binds `public_value == trace_cell == witness_value`. There is no
//! permutation and no per-op params.

use alloc::boxed::Box;
use alloc::format;
use alloc::string::ToString;
use alloc::vec::Vec;
use core::any::Any;
use core::fmt::Debug;
use core::marker::PhantomData;

use p3_field::{BasedVectorSpace, ExtensionField, Field, PrimeField64};

use crate::CircuitError;
use crate::builder::{CircuitBuilderError, NpoCircuitPlugin, NpoLoweringContext};
use crate::ops::{ExecutionContext, NonPrimitiveExecutor, NpoTypeId, Op, PreprocessedWriter};
use crate::tables::{NonPrimitiveTrace, TraceGeneratorFn};
use crate::types::{ExprId, WitnessId};

// ============================================================================
// Execution State
// ============================================================================

/// Per-op data captured during execution: the K input witnesses and values.
#[derive(Debug, Clone)]
pub struct ExposeCircuitRow<F> {
    pub input_wids: Vec<WitnessId>,
    pub input_values: Vec<F>,
}

/// Execution state collecting rows across all expose calls.
#[derive(Debug, Default)]
pub struct ExposeExecutionState<F> {
    pub rows: Vec<ExposeCircuitRow<F>>,
}

// ============================================================================
// Executor
// ============================================================================

/// Executor for a single expose op. No per-op params, no outputs.
pub struct ExposeExecutor<F> {
    op_type: NpoTypeId,
    _phantom: PhantomData<F>,
}

impl<F> Clone for ExposeExecutor<F> {
    fn clone(&self) -> Self {
        Self { op_type: self.op_type.clone(), _phantom: PhantomData }
    }
}

impl<F> Debug for ExposeExecutor<F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ExposeExecutor").field("op_type", &self.op_type).finish()
    }
}

impl<F> Default for ExposeExecutor<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F> ExposeExecutor<F> {
    pub fn new() -> Self {
        Self { op_type: NpoTypeId::expose(), _phantom: PhantomData }
    }
}

impl<F: Field + Send + Sync + 'static> NonPrimitiveExecutor<F> for ExposeExecutor<F> {
    fn execute(
        &self,
        inputs: &[Vec<WitnessId>],
        outputs: &[Vec<WitnessId>],
        ctx: &mut ExecutionContext<'_, F>,
    ) -> Result<(), CircuitError> {
        if inputs.len() != 1 || !outputs.is_empty() {
            return Err(CircuitError::NonPrimitiveOpLayoutMismatch {
                op: self.op_type.clone(),
                expected: "1 input group and 0 output groups".to_string(),
                got: inputs.len() + outputs.len(),
            });
        }
        let input_wids = &inputs[0];
        let mut input_values = Vec::with_capacity(input_wids.len());
        for &wid in input_wids {
            input_values.push(ctx.get_witness(wid)?);
        }

        let state = ctx.get_op_state_mut::<ExposeExecutionState<F>>(&self.op_type);
        state.rows.push(ExposeCircuitRow { input_wids: input_wids.clone(), input_values });
        Ok(())
    }

    fn op_type(&self) -> &NpoTypeId {
        &self.op_type
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    /// Preprocessed layout: 2 columns per exposed lane.
    /// - col 0 `input_idx` — input witness base-field index (bumps `ext_reads[input_wid]`)
    /// - col 1 `is_real`   — `F::ONE` for real rows, `F::ZERO` for padding (the
    ///   `WitnessChecks` read multiplier and the public-value-binding gate)
    fn preprocess(
        &self,
        inputs: &[Vec<WitnessId>],
        _outputs: &[Vec<WitnessId>],
        preprocessed: &mut dyn PreprocessedWriter<F>,
    ) -> Result<(), CircuitError> {
        let input_wids = &inputs[0];
        for &in_wid in input_wids {
            // col 0: input_idx (+ bumps ext_reads)
            preprocessed.register_non_primitive_witness_reads(&self.op_type, &[in_wid])?;
            // col 1: is_real = 1
            preprocessed.register_non_primitive_preprocessed_no_read(&self.op_type, &[F::ONE]);
        }
        Ok(())
    }

    fn num_exposed_outputs(&self) -> Option<usize> {
        Some(0)
    }

    fn boxed(&self) -> Box<dyn NonPrimitiveExecutor<F>> {
        Box::new(self.clone())
    }
}

// ============================================================================
// Circuit Plugin
// ============================================================================

/// Circuit-layer plugin for expose non-primitive operations. Holds the trace
/// generator (`generate_expose_trace::<BF, EF>`).
pub struct ExposeCircuitPlugin<F> {
    trace_gen: TraceGeneratorFn<F>,
    _phantom: PhantomData<F>,
}

impl<F> ExposeCircuitPlugin<F> {
    pub fn new(trace_gen: TraceGeneratorFn<F>) -> Self {
        Self { trace_gen, _phantom: PhantomData }
    }
}

impl<F> Debug for ExposeCircuitPlugin<F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("ExposeCircuitPlugin")
    }
}

// SAFETY: function pointers + PhantomData<F> are Send + Sync; we store no F values.
unsafe impl<F> Send for ExposeCircuitPlugin<F> {}
unsafe impl<F> Sync for ExposeCircuitPlugin<F> {}

impl<F> NpoCircuitPlugin<F> for ExposeCircuitPlugin<F>
where
    F: Field + Send + Sync + 'static,
{
    fn type_id(&self) -> NpoTypeId {
        NpoTypeId::expose()
    }

    fn lower(
        &self,
        data: &crate::builder::NonPrimitiveOperationData<F>,
        output_exprs: &[(u32, ExprId)],
        ctx: &mut NpoLoweringContext<'_, F>,
    ) -> Result<Op<F>, CircuitBuilderError> {
        if data.input_exprs.len() != 1 {
            return Err(CircuitBuilderError::NonPrimitiveOpArity {
                op: "Expose",
                expected: "1 input group".to_string(),
                got: data.input_exprs.len(),
            });
        }
        if !output_exprs.is_empty() {
            return Err(CircuitBuilderError::NonPrimitiveOpArity {
                op: "Expose",
                expected: "0 outputs".to_string(),
                got: output_exprs.len(),
            });
        }
        let input_wids: Vec<WitnessId> = data.input_exprs[0]
            .iter()
            .enumerate()
            .map(|(i, &expr)| ctx.resolve_witness_id(expr, &format!("Expose input {i}")))
            .collect::<Result<_, _>>()?;

        Ok(Op::NonPrimitiveOpWithExecutor {
            inputs: alloc::vec![input_wids],
            outputs: alloc::vec![],
            executor: Box::new(ExposeExecutor::<F>::new()),
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

/// Trace for expose operations.
#[derive(Debug, Clone)]
pub struct ExposeTrace<F> {
    pub operations: Vec<ExposeCircuitRow<F>>,
}

impl<F> ExposeTrace<F> {
    /// Total number of exposed lanes across all collected ops.
    pub fn total_rows(&self) -> usize {
        self.operations.iter().map(|r| r.input_wids.len()).sum()
    }
}

impl<TraceF: Clone + Send + Sync + Debug + 'static, CF> NonPrimitiveTrace<CF> for ExposeTrace<TraceF> {
    fn op_type(&self) -> NpoTypeId {
        NpoTypeId::expose()
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

/// Generate the expose trace from execution state, unpacking each EF value into
/// its `D` base-field coefficients (mirrors `generate_permute_trace`).
pub fn generate_expose_trace<BF, EF>(
    op_states: &crate::ops::OpStateMap,
) -> Result<Option<Box<dyn NonPrimitiveTrace<EF>>>, CircuitError>
where
    BF: PrimeField64,
    EF: ExtensionField<BF>,
{
    let Some(state) = op_states
        .get(&NpoTypeId::expose())
        .and_then(|s| s.downcast_ref::<ExposeExecutionState<EF>>())
    else {
        return Ok(None);
    };

    if state.rows.is_empty() {
        return Ok(None);
    }

    let operations: Vec<ExposeCircuitRow<BF>> = state
        .rows
        .iter()
        .map(|row| {
            let input_values: Vec<BF> = row
                .input_values
                .iter()
                .flat_map(|ef| {
                    <EF as BasedVectorSpace<BF>>::as_basis_coefficients_slice(ef)
                        .iter()
                        .copied()
                })
                .collect();
            ExposeCircuitRow { input_wids: row.input_wids.clone(), input_values }
        })
        .collect();

    Ok(Some(Box::new(ExposeTrace { operations })))
}
