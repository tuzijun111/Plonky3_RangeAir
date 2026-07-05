//! Range non-primitive operation: proves each input witness value lies in `[0, 256)` via a
//! LogUp lookup against a fixed 256-row byte table on a dedicated `RangeBus`.
//!
//! Unlike [`super::permute`], this op has **1 input group and 0 output groups** — it asserts a
//! property of existing witnesses rather than producing new ones. Each `add_range` call emits one
//! *lookup row* per input value:
//!
//! - The shared `WitnessChecks` bus binds the row's `value` main column to the witness id (as a
//!   reader), so the looked-up value is exactly the circuit witness.
//! - A dedicated `RangeBus` sends `+1` per lookup value.
//!
//! The fixed byte table (256 rows, `table_const = 0..=255`) is NOT emitted here — it is appended
//! once, globally, by `RangePreprocessor` (preprocessed side) and `RangeAir::trace_to_matrix`
//! (main side, carrying the per-byte multiplicity witness). Balance of `RangeBus` across all rows
//! holds iff every looked-up value equals some table byte, i.e. lies in `[0, 256)`.
//!
//! Wide values (e.g. 22-bit GK errors) are range-checked by first decomposing into bytes at the
//! builder layer, then range-checking each byte here; that decomposition is added on top.

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

/// Per-op data captured during execution: the witness ids checked and their values.
#[derive(Debug, Clone)]
pub struct RangeCircuitRow<F> {
    pub input_wids: Vec<WitnessId>,
    /// Value coefficients: `D` consecutive base-field elements per logical checked value.
    pub input_values: Vec<F>,
}

/// Execution state collecting rows across all range calls.
#[derive(Debug, Default)]
pub struct RangeExecutionState<F> {
    pub rows: Vec<RangeCircuitRow<F>>,
}

// ============================================================================
// Executor
// ============================================================================

/// Executor for a single range op. Records the checked witness values; sets no outputs.
pub struct RangeExecutor<F> {
    op_type: NpoTypeId,
    _phantom: PhantomData<F>,
}

impl<F> Clone for RangeExecutor<F> {
    fn clone(&self) -> Self {
        Self {
            op_type: self.op_type.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<F> Debug for RangeExecutor<F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RangeExecutor")
            .field("op_type", &self.op_type)
            .finish()
    }
}

impl<F> RangeExecutor<F> {
    pub fn new() -> Self {
        Self {
            op_type: NpoTypeId::range(),
            _phantom: PhantomData,
        }
    }
}

impl<F> Default for RangeExecutor<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Field + Send + Sync + 'static> NonPrimitiveExecutor<F> for RangeExecutor<F> {
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

        let state = ctx.get_op_state_mut::<RangeExecutionState<F>>(&self.op_type);
        state.rows.push(RangeCircuitRow {
            input_wids: input_wids.clone(),
            input_values,
        });

        Ok(())
    }

    fn op_type(&self) -> &NpoTypeId {
        &self.op_type
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    /// Preprocessed layout: 4 columns per checked value (one lookup row each).
    ///
    /// - col 0 `witness_idx` — checked witness index (also bumps `ext_reads[wid]`)
    /// - col 1 `is_lookup`   — constant `F::ONE`
    /// - col 2 `table_const` — `F::ZERO` (unused on lookup rows)
    /// - col 3 `is_table`    — `F::ZERO`
    ///
    /// The 256 table rows (`is_table = 1`, `table_const = 0..=255`) are appended globally by
    /// `RangePreprocessor`, not here.
    fn preprocess(
        &self,
        inputs: &[Vec<WitnessId>],
        _outputs: &[Vec<WitnessId>],
        preprocessed: &mut dyn PreprocessedWriter<F>,
    ) -> Result<(), CircuitError> {
        let input_wids = &inputs[0];
        for &wid in input_wids {
            // col 0: witness_idx (+ bumps ext_reads so the value's creator balances the read)
            preprocessed.register_non_primitive_witness_reads(&self.op_type, &[wid])?;
            // col 1: is_lookup = 1
            preprocessed.register_non_primitive_preprocessed_no_read(&self.op_type, &[F::ONE]);
            // col 2: table_const = 0
            preprocessed.register_non_primitive_preprocessed_no_read(&self.op_type, &[F::ZERO]);
            // col 3: is_table = 0
            preprocessed.register_non_primitive_preprocessed_no_read(&self.op_type, &[F::ZERO]);
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

/// Circuit-layer plugin for range non-primitive operations.
pub struct RangeCircuitPlugin<F> {
    trace_gen: TraceGeneratorFn<F>,
    _phantom: PhantomData<F>,
}

impl<F> RangeCircuitPlugin<F> {
    pub fn new(trace_gen: TraceGeneratorFn<F>) -> Self {
        Self {
            trace_gen,
            _phantom: PhantomData,
        }
    }
}

impl<F> Debug for RangeCircuitPlugin<F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("RangeCircuitPlugin")
    }
}

// SAFETY: function pointers are `Send + Sync`, and `PhantomData<F>` stores no `F`.
unsafe impl<F> Send for RangeCircuitPlugin<F> {}
unsafe impl<F> Sync for RangeCircuitPlugin<F> {}

impl<F> NpoCircuitPlugin<F> for RangeCircuitPlugin<F>
where
    F: Field + Send + Sync + 'static,
{
    fn type_id(&self) -> NpoTypeId {
        NpoTypeId::range()
    }

    fn lower(
        &self,
        data: &crate::builder::NonPrimitiveOperationData<F>,
        output_exprs: &[(u32, ExprId)],
        ctx: &mut NpoLoweringContext<'_, F>,
    ) -> Result<Op<F>, CircuitBuilderError> {
        if data.input_exprs.len() != 1 {
            return Err(CircuitBuilderError::NonPrimitiveOpArity {
                op: "Range",
                expected: "1 input group".to_string(),
                got: data.input_exprs.len(),
            });
        }
        if !output_exprs.is_empty() {
            return Err(CircuitBuilderError::NonPrimitiveOpArity {
                op: "Range",
                expected: "0 outputs".to_string(),
                got: output_exprs.len(),
            });
        }

        let input_wids: Vec<WitnessId> = data.input_exprs[0]
            .iter()
            .enumerate()
            .map(|(i, &expr)| ctx.resolve_witness_id(expr, &format!("Range input {i}")))
            .collect::<Result<_, _>>()?;

        Ok(Op::NonPrimitiveOpWithExecutor {
            inputs: alloc::vec![input_wids],
            outputs: alloc::vec![],
            executor: Box::new(RangeExecutor::<F>::new()),
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

/// Trace for range operations.
#[derive(Debug, Clone)]
pub struct RangeTrace<F> {
    pub operations: Vec<RangeCircuitRow<F>>,
}

impl<F> RangeTrace<F> {
    /// Total number of looked-up values (lookup rows) across all collected ops.
    pub fn total_lookups(&self) -> usize {
        self.operations.iter().map(|r| r.input_wids.len()).sum()
    }
}

impl<TraceF: Clone + Send + Sync + Debug + 'static, CF> NonPrimitiveTrace<CF> for RangeTrace<TraceF> {
    fn op_type(&self) -> NpoTypeId {
        NpoTypeId::range()
    }

    fn rows(&self) -> usize {
        self.total_lookups()
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

/// Generate the range trace from execution state, unpacking each `EF` value into its `D`
/// base-field coefficients (identity for `EF == BF`, D=1).
pub fn generate_range_trace<BF, EF>(
    op_states: &crate::ops::OpStateMap,
) -> Result<Option<Box<dyn NonPrimitiveTrace<EF>>>, CircuitError>
where
    BF: PrimeField64,
    EF: ExtensionField<BF>,
{
    let Some(state) = op_states
        .get(&NpoTypeId::range())
        .and_then(|s| s.downcast_ref::<RangeExecutionState<EF>>())
    else {
        return Ok(None);
    };

    if state.rows.is_empty() {
        return Ok(None);
    }

    let operations: Vec<RangeCircuitRow<BF>> = state
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
            RangeCircuitRow {
                input_wids: row.input_wids.clone(),
                input_values,
            }
        })
        .collect();

    Ok(Some(Box::new(RangeTrace { operations })))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use p3_field::PrimeCharacteristicRing;
    use p3_test_utils::baby_bear_params::BabyBear;

    use super::*;
    use crate::builder::CircuitBuilder;

    type F = BabyBear;

    /// CircuitRunner-only smoke test: the executor records checked values, sets no outputs.
    #[test]
    fn range_executor_records_values() {
        let mut builder = CircuitBuilder::<F>::new();
        builder.enable_range(generate_range_trace::<F, F>);

        let inputs: Vec<ExprId> = (0..3).map(|_| builder.public_input()).collect();
        builder.add_range(&inputs).expect("add_range accepts inputs");

        let circuit = builder.build().expect("build range circuit");
        let mut runner = circuit.runner();
        runner
            .set_public_inputs(&[F::from_u64(7), F::from_u64(42), F::from_u64(255)])
            .expect("set public inputs");
        runner.run().expect("circuit runner executes range op");
    }
}
