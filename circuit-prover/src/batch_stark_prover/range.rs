//! Range table prover: builds `RangeAir` instances (lookup rows + the fixed 256-row byte table).
//!
//! The lookup rows' preprocessed data (`witness_idx`, `is_lookup`) is emitted by the op's
//! `preprocess`; `RangePreprocessor` appends the 256 table rows (`table_const = 0..=255`,
//! `is_table = 1`). `RangeProver::batch_instance_base` builds the aligned main trace (lookup values
//! + per-byte multiplicity witnesses) via `RangeAir::trace_to_matrix`.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use hashbrown::HashMap;
use p3_baby_bear::BabyBear;
use p3_batch_stark::{StarkGenericConfig, Val};
use p3_circuit::ops::range::RangeTrace;
use p3_circuit::ops::{NonPrimitivePreprocessedMap, NpoTypeId};
use p3_circuit::tables::Traces;
use p3_circuit::{CircuitError, PreprocessedColumns};
use p3_field::extension::{BinomialExtensionField, QuinticTrinomialExtensionField};
use p3_field::{Algebra, ExtensionField, Field, PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;
use p3_koala_bear::KoalaBear;
use p3_uni_stark::{SymbolicExpression, SymbolicExpressionExt};
use p3_util::log2_ceil_usize;

use super::dynamic_air::{
    BatchAir, BatchTableInstance, DynamicAirEntry, TableProver, transmute_traces,
};
use super::{NonPrimitiveTableEntry, TablePacking};
use crate::air::RangeAir;
use crate::air::range_air::{
    RANGE_PREP_IS_LOOKUP, RANGE_PREP_IS_TABLE, RANGE_PREP_LANE_WIDTH, RANGE_PREP_TABLE_CONST,
    RANGE_PREP_WITNESS_IDX, RANGE_TABLE_SIZE,
};
use crate::common::{CircuitTableAir, NpoAirBuilder, NpoPreprocessor};
use crate::config::StarkField;
use crate::{ConstraintProfile, impl_table_prover_batch_instances_from_base};

impl<SC, const D: usize> BatchAir<SC> for RangeAir<Val<SC>, D>
where
    SC: StarkGenericConfig + Send + Sync,
    Val<SC>: StarkField,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
{
}

/// Table prover for the range-check NPO. Pinned to `lanes = 1`.
pub struct RangeProver<const D: usize> {
    lanes: usize,
}

impl<const D: usize> RangeProver<D> {
    pub fn new(lanes: usize) -> Self {
        Self {
            lanes: lanes.max(1),
        }
    }

    fn batch_instance_base<SC>(
        &self,
        _config: &SC,
        packing: &TablePacking,
        traces: &Traces<Val<SC>>,
    ) -> Option<BatchTableInstance<SC>>
    where
        SC: StarkGenericConfig + 'static + Send + Sync,
        Val<SC>: StarkField,
        SymbolicExpressionExt<Val<SC>, SC::Challenge>:
            Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
    {
        let op_type = NpoTypeId::range();
        let trace = traces.non_primitive_traces.get(&op_type)?;
        if trace.rows() == 0 {
            return None;
        }

        let t = trace.as_any().downcast_ref::<RangeTrace<Val<SC>>>()?;

        let num_lookups = t.total_lookups();
        let num_rows = num_lookups + RANGE_TABLE_SIZE;
        let lanes = packing.npo_lanes(&op_type).unwrap_or(self.lanes);
        let min_height = packing.min_trace_height();

        // Placeholder preprocessed (overridden by the committed columns before proving); built to
        // match the committed layout so the debug lookup check is self-consistent.
        let prep_lane_width = RangeAir::<Val<SC>, D>::preprocessed_lane_width();
        let mut preprocessed = Val::<SC>::zero_vec(num_rows * prep_lane_width);
        let mut entry_idx = 0usize;
        for op in t.operations.iter() {
            for &wid in op.input_wids.iter() {
                let base = entry_idx * prep_lane_width;
                preprocessed[base + RANGE_PREP_WITNESS_IDX] = wid.base_field_index::<Val<SC>, D>();
                preprocessed[base + RANGE_PREP_IS_LOOKUP] = Val::<SC>::ONE;
                // table_const, is_table stay zero on lookup rows.
                entry_idx += 1;
            }
        }
        for b in 0..RANGE_TABLE_SIZE {
            let base = (num_lookups + b) * prep_lane_width;
            preprocessed[base + RANGE_PREP_TABLE_CONST] = Val::<SC>::from_u64(b as u64);
            preprocessed[base + RANGE_PREP_IS_TABLE] = Val::<SC>::ONE;
        }

        let air = RangeAir::<Val<SC>, D>::new_with_preprocessed(lanes, preprocessed, min_height);
        let matrix = RangeAir::<Val<SC>, D>::trace_to_matrix(&t.operations, lanes);

        Some(BatchTableInstance {
            op_type,
            air: DynamicAirEntry::new(Box::new(air)),
            trace: matrix,
            public_values: Vec::new(),
            rows: num_rows,
            lanes,
        })
    }
}

impl<SC, const D: usize> TableProver<SC> for RangeProver<D>
where
    SC: StarkGenericConfig + 'static + Send + Sync,
    Val<SC>: StarkField,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
{
    fn op_type(&self) -> NpoTypeId {
        NpoTypeId::range()
    }

    fn lanes(&self) -> usize {
        self.lanes
    }

    impl_table_prover_batch_instances_from_base!(batch_instance_base);

    fn batch_air_from_table_entry(
        &self,
        _config: &SC,
        _degree: usize,
        _circuit_extension_degree: u32,
        table_entry: &NonPrimitiveTableEntry<SC>,
    ) -> Result<DynamicAirEntry<SC>, String> {
        let air = RangeAir::<Val<SC>, D>::new_with_preprocessed(table_entry.lanes, Vec::new(), 1);
        Ok(DynamicAirEntry::new(Box::new(air)))
    }

    fn air_with_committed_preprocessed(
        &self,
        committed_prep: Vec<Val<SC>>,
        min_height: usize,
        lanes: usize,
        _circuit_extension_degree: u32,
    ) -> Option<DynamicAirEntry<SC>> {
        let air = RangeAir::<Val<SC>, D>::new_with_preprocessed(lanes, committed_prep, min_height);
        Some(DynamicAirEntry::new(Box::new(air)))
    }
}

// ============================================================================
// Preprocessor — converts the op's EF lookup rows to base and appends the 256-row byte table.
// ============================================================================

/// NpoPreprocessor for the range table.
#[derive(Default, Clone)]
pub struct RangePreprocessor;

impl RangePreprocessor {
    pub const fn new() -> Self {
        Self
    }
}

impl NpoPreprocessor<KoalaBear> for RangePreprocessor {
    fn preprocess(
        &self,
        _circuit: &dyn core::any::Any,
        preprocessed: &mut dyn core::any::Any,
    ) -> Result<NonPrimitivePreprocessedMap<KoalaBear>, CircuitError> {
        type F = KoalaBear;
        if let Some(prep) =
            preprocessed.downcast_mut::<PreprocessedColumns<BinomialExtensionField<F, 4>, 4>>()
        {
            return range_preprocess_impl::<F, _, 4>(prep);
        }
        if let Some(prep) =
            preprocessed.downcast_mut::<PreprocessedColumns<QuinticTrinomialExtensionField<F>, 5>>()
        {
            return range_preprocess_impl::<F, _, 5>(prep);
        }
        if let Some(prep) = preprocessed.downcast_mut::<PreprocessedColumns<F, 1>>() {
            return range_preprocess_impl::<F, _, 1>(prep);
        }
        Ok(HashMap::new())
    }
}

impl NpoPreprocessor<BabyBear> for RangePreprocessor {
    fn preprocess(
        &self,
        _circuit: &dyn core::any::Any,
        preprocessed: &mut dyn core::any::Any,
    ) -> Result<NonPrimitivePreprocessedMap<BabyBear>, CircuitError> {
        type F = BabyBear;
        if let Some(prep) =
            preprocessed.downcast_mut::<PreprocessedColumns<BinomialExtensionField<F, 4>, 4>>()
        {
            return range_preprocess_impl::<F, _, 4>(prep);
        }
        if let Some(prep) = preprocessed.downcast_mut::<PreprocessedColumns<F, 1>>() {
            return range_preprocess_impl::<F, _, 1>(prep);
        }
        Ok(HashMap::new())
    }
}

impl NpoPreprocessor<Goldilocks> for RangePreprocessor {
    fn preprocess(
        &self,
        _circuit: &dyn core::any::Any,
        preprocessed: &mut dyn core::any::Any,
    ) -> Result<NonPrimitivePreprocessedMap<Goldilocks>, CircuitError> {
        type F = Goldilocks;
        if let Some(prep) =
            preprocessed.downcast_mut::<PreprocessedColumns<BinomialExtensionField<F, 2>, 2>>()
        {
            return range_preprocess_impl::<F, _, 2>(prep);
        }
        if let Some(prep) = preprocessed.downcast_mut::<PreprocessedColumns<F, 1>>() {
            return range_preprocess_impl::<F, _, 1>(prep);
        }
        Ok(HashMap::new())
    }
}

fn range_preprocess_impl<F, EF, const D: usize>(
    prep: &PreprocessedColumns<EF, D>,
) -> Result<NonPrimitivePreprocessedMap<F>, CircuitError>
where
    F: StarkField + PrimeField64,
    EF: Field + ExtensionField<F> + 'static,
{
    let op_type = NpoTypeId::range();
    let ef_data = match prep.non_primitive.get(&op_type) {
        Some(d) if !d.is_empty() => d,
        _ => return Ok(HashMap::new()),
    };

    // Convert the op-emitted lookup rows from EF to base field.
    let mut prep_base: Vec<F> = ef_data
        .iter()
        .map(|v| v.as_base().ok_or(CircuitError::InvalidPreprocessedValues))
        .collect::<Result<Vec<_>, CircuitError>>()?;

    if !prep_base.len().is_multiple_of(RANGE_PREP_LANE_WIDTH) {
        return Err(CircuitError::InvalidPreprocessedValues);
    }

    // Append the fixed 256-row byte table: [witness_idx=0, is_lookup=0, table_const=b, is_table=1].
    prep_base.reserve(RANGE_TABLE_SIZE * RANGE_PREP_LANE_WIDTH);
    for b in 0..RANGE_TABLE_SIZE {
        let mut row = [F::ZERO; RANGE_PREP_LANE_WIDTH];
        row[RANGE_PREP_TABLE_CONST] = F::from_u64(b as u64);
        row[RANGE_PREP_IS_TABLE] = F::ONE;
        prep_base.extend_from_slice(&row);
    }

    let mut result = HashMap::new();
    result.insert(op_type, prep_base);
    Ok(result)
}

// ============================================================================
// AIR Builder
// ============================================================================

/// NpoAirBuilder for the range table.
#[derive(Clone)]
pub struct RangeAirBuilder<const D: usize> {
    lanes: usize,
}

impl<const D: usize> RangeAirBuilder<D> {
    pub fn new(lanes: usize) -> Self {
        Self {
            lanes: lanes.max(1),
        }
    }
}

impl<SC, const D: usize> NpoAirBuilder<SC, D> for RangeAirBuilder<D>
where
    SC: StarkGenericConfig + 'static + Send + Sync,
    Val<SC>: StarkField,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
{
    fn lanes(&self) -> usize {
        self.lanes
    }

    fn try_build(
        &self,
        op_type: &NpoTypeId,
        prep_base: &[Val<SC>],
        min_height: usize,
        lanes: usize,
        _constraint_profile: ConstraintProfile,
    ) -> Option<(CircuitTableAir<SC, D>, usize)> {
        if op_type.as_str() != "range" {
            return None;
        }

        let num_entries = prep_base.len() / RANGE_PREP_LANE_WIDTH;
        let num_rows = num_entries.div_ceil(lanes).max(1);

        let air =
            RangeAir::<Val<SC>, D>::new_with_preprocessed(lanes, prep_base.to_vec(), min_height);

        let padded_rows = num_rows
            .next_power_of_two()
            .max(min_height.next_power_of_two());
        let degree = log2_ceil_usize(padded_rows);

        Some((
            CircuitTableAir::Dynamic(DynamicAirEntry::new(Box::new(air))),
            degree,
        ))
    }
}
