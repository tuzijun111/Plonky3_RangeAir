//! Expose table prover: builds `ExposeAir` instances and — uniquely among the
//! NPOs — sets `BatchTableInstance.public_values` to the exposed witness
//! values, so they surface as recursion `air_public_targets`.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use hashbrown::HashMap;
use p3_baby_bear::BabyBear;
use p3_batch_stark::{StarkGenericConfig, Val};
use p3_circuit::ops::expose::ExposeTrace;
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
use crate::air::ExposeAir;
use crate::air::expose_air::{EXPOSE_PREP_IN_IDX, EXPOSE_PREP_IS_REAL, EXPOSE_PREP_LANE_WIDTH};
use crate::common::{CircuitTableAir, NpoAirBuilder, NpoPreprocessor};
use crate::config::StarkField;
use crate::{ConstraintProfile, impl_table_prover_batch_instances_from_base};

impl<SC, const D: usize> BatchAir<SC> for ExposeAir<Val<SC>, D>
where
    SC: StarkGenericConfig + Send + Sync,
    Val<SC>: StarkField,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
{
}

/// Table prover for the expose (public-value surfacing) NPO.
pub struct ExposeProver<const D: usize>;

impl<const D: usize> Default for ExposeProver<D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const D: usize> ExposeProver<D> {
    pub fn new() -> Self {
        Self
    }

    fn batch_instance_base<SC>(
        &self,
        _config: &SC,
        _packing: &TablePacking,
        traces: &Traces<Val<SC>>,
    ) -> Option<BatchTableInstance<SC>>
    where
        SC: StarkGenericConfig + 'static + Send + Sync,
        Val<SC>: StarkField,
        SymbolicExpressionExt<Val<SC>, SC::Challenge>:
            Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
    {
        let op_type = NpoTypeId::expose();
        let trace = traces.non_primitive_traces.get(&op_type)?;
        if trace.rows() == 0 {
            return None;
        }

        let t = trace.as_any().downcast_ref::<ExposeTrace<Val<SC>>>()?;

        // Pack all exposed lanes into a single real row so the global public
        // values line up with row 0's lanes.
        let total: usize = t.total_rows();
        let lanes = total.max(1);
        let min_height = _packing.min_trace_height();

        let prep_lane_width = ExposeAir::<Val<SC>, D>::preprocessed_lane_width();
        let mut preprocessed = Val::<SC>::zero_vec(lanes * prep_lane_width);
        let mut public_values: Vec<Val<SC>> = Vec::with_capacity(total * D);
        let mut lane_idx = 0usize;
        for op in t.operations.iter() {
            for (slot, &in_wid) in op.input_wids.iter().enumerate() {
                let base = lane_idx * prep_lane_width;
                preprocessed[base + EXPOSE_PREP_IN_IDX] = in_wid.base_field_index::<Val<SC>, D>();
                preprocessed[base + EXPOSE_PREP_IS_REAL] = Val::<SC>::ONE;
                for j in 0..D {
                    public_values.push(op.input_values[slot * D + j]);
                }
                lane_idx += 1;
            }
        }

        let air = ExposeAir::<Val<SC>, D>::new_with_preprocessed(lanes, preprocessed, min_height);
        let matrix = ExposeAir::<Val<SC>, D>::trace_to_matrix(&t.operations, lanes);

        Some(BatchTableInstance {
            op_type,
            air: DynamicAirEntry::new(Box::new(air)),
            trace: matrix,
            public_values,
            rows: total,
            lanes,
        })
    }
}

impl<SC, const D: usize> TableProver<SC> for ExposeProver<D>
where
    SC: StarkGenericConfig + 'static + Send + Sync,
    Val<SC>: StarkField,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
{
    fn op_type(&self) -> NpoTypeId {
        NpoTypeId::expose()
    }

    fn lanes(&self) -> usize {
        1
    }

    impl_table_prover_batch_instances_from_base!(batch_instance_base);

    fn batch_air_from_table_entry(
        &self,
        _config: &SC,
        _degree: usize,
        _circuit_extension_degree: u32,
        table_entry: &NonPrimitiveTableEntry<SC>,
    ) -> Result<DynamicAirEntry<SC>, String> {
        let air = ExposeAir::<Val<SC>, D>::new_with_preprocessed(table_entry.lanes, Vec::new(), 1);
        Ok(DynamicAirEntry::new(Box::new(air)))
    }

    fn air_with_committed_preprocessed(
        &self,
        committed_prep: Vec<Val<SC>>,
        min_height: usize,
        lanes: usize,
        _circuit_extension_degree: u32,
    ) -> Option<DynamicAirEntry<SC>> {
        let air = ExposeAir::<Val<SC>, D>::new_with_preprocessed(lanes, committed_prep, min_height);
        Some(DynamicAirEntry::new(Box::new(air)))
    }
}

// ============================================================================
// Preprocessor — converts committed EF preprocessed to BF (no patching needed:
// in_idx / is_real do not depend on ext_reads).
// ============================================================================

#[derive(Default, Clone)]
pub struct ExposePreprocessor;

impl ExposePreprocessor {
    pub const fn new() -> Self {
        Self
    }
}

impl NpoPreprocessor<KoalaBear> for ExposePreprocessor {
    fn preprocess(
        &self,
        _circuit: &dyn core::any::Any,
        preprocessed: &mut dyn core::any::Any,
    ) -> Result<NonPrimitivePreprocessedMap<KoalaBear>, CircuitError> {
        type F = KoalaBear;
        if let Some(prep) =
            preprocessed.downcast_mut::<PreprocessedColumns<BinomialExtensionField<F, 4>, 4>>()
        {
            return expose_preprocess_impl::<F, _, 4>(prep);
        }
        if let Some(prep) =
            preprocessed.downcast_mut::<PreprocessedColumns<QuinticTrinomialExtensionField<F>, 5>>()
        {
            return expose_preprocess_impl::<F, _, 5>(prep);
        }
        if let Some(prep) = preprocessed.downcast_mut::<PreprocessedColumns<F, 1>>() {
            return expose_preprocess_impl::<F, _, 1>(prep);
        }
        Ok(HashMap::new())
    }
}

impl NpoPreprocessor<BabyBear> for ExposePreprocessor {
    fn preprocess(
        &self,
        _circuit: &dyn core::any::Any,
        preprocessed: &mut dyn core::any::Any,
    ) -> Result<NonPrimitivePreprocessedMap<BabyBear>, CircuitError> {
        type F = BabyBear;
        if let Some(prep) =
            preprocessed.downcast_mut::<PreprocessedColumns<BinomialExtensionField<F, 4>, 4>>()
        {
            return expose_preprocess_impl::<F, _, 4>(prep);
        }
        if let Some(prep) = preprocessed.downcast_mut::<PreprocessedColumns<F, 1>>() {
            return expose_preprocess_impl::<F, _, 1>(prep);
        }
        Ok(HashMap::new())
    }
}

impl NpoPreprocessor<Goldilocks> for ExposePreprocessor {
    fn preprocess(
        &self,
        _circuit: &dyn core::any::Any,
        preprocessed: &mut dyn core::any::Any,
    ) -> Result<NonPrimitivePreprocessedMap<Goldilocks>, CircuitError> {
        type F = Goldilocks;
        if let Some(prep) =
            preprocessed.downcast_mut::<PreprocessedColumns<BinomialExtensionField<F, 2>, 2>>()
        {
            return expose_preprocess_impl::<F, _, 2>(prep);
        }
        if let Some(prep) = preprocessed.downcast_mut::<PreprocessedColumns<F, 1>>() {
            return expose_preprocess_impl::<F, _, 1>(prep);
        }
        Ok(HashMap::new())
    }
}

fn expose_preprocess_impl<F, EF, const D: usize>(
    prep: &PreprocessedColumns<EF, D>,
) -> Result<NonPrimitivePreprocessedMap<F>, CircuitError>
where
    F: StarkField + PrimeField64,
    EF: Field + ExtensionField<F> + 'static,
{
    let op_type = NpoTypeId::expose();
    let ef_data = match prep.non_primitive.get(&op_type) {
        Some(d) if !d.is_empty() => d,
        _ => return Ok(HashMap::new()),
    };

    let prep_base: Vec<F> = ef_data
        .iter()
        .map(|v| v.as_base().ok_or(CircuitError::InvalidPreprocessedValues))
        .collect::<Result<Vec<_>, CircuitError>>()?;

    if !prep_base.len().is_multiple_of(EXPOSE_PREP_LANE_WIDTH) {
        return Err(CircuitError::InvalidPreprocessedValues);
    }

    let mut result = HashMap::new();
    result.insert(op_type, prep_base);
    Ok(result)
}

// ============================================================================
// AIR Builder
// ============================================================================

#[derive(Clone, Default)]
pub struct ExposeAirBuilder<const D: usize>;

impl<const D: usize> ExposeAirBuilder<D> {
    pub fn new() -> Self {
        Self
    }
}

impl<SC, const D: usize> NpoAirBuilder<SC, D> for ExposeAirBuilder<D>
where
    SC: StarkGenericConfig + 'static + Send + Sync,
    Val<SC>: StarkField,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
{
    fn lanes(&self) -> usize {
        1
    }

    fn try_build(
        &self,
        op_type: &NpoTypeId,
        prep_base: &[Val<SC>],
        min_height: usize,
        lanes: usize,
        _constraint_profile: ConstraintProfile,
    ) -> Option<(CircuitTableAir<SC, D>, usize)> {
        if op_type.as_str() != "expose" {
            return None;
        }

        let num_lanes = prep_base.len() / EXPOSE_PREP_LANE_WIDTH;
        let num_rows = num_lanes.div_ceil(lanes.max(1)).max(1);

        let air = ExposeAir::<Val<SC>, D>::new_with_preprocessed(lanes, prep_base.to_vec(), min_height);

        let padded_rows = num_rows.next_power_of_two().max(min_height.next_power_of_two());
        let degree = log2_ceil_usize(padded_rows);

        Some((CircuitTableAir::Dynamic(DynamicAirEntry::new(Box::new(air))), degree))
    }
}
