//! Permute table prover: builds `PermuteAir` instances + patches `out_mult` from `ext_reads`.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use hashbrown::HashMap;
use p3_baby_bear::BabyBear;
use p3_batch_stark::{StarkGenericConfig, Val};
use p3_circuit::ops::permute::PermuteTrace;
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
use crate::air::PermuteAir;
use crate::air::permute_air::{
    PERMUTE_PREP_IN_IDX, PERMUTE_PREP_IN_MULT, PERMUTE_PREP_IS_REAL, PERMUTE_PREP_LANE_WIDTH,
    PERMUTE_PREP_OUT_IDX, PERMUTE_PREP_OUT_MULT,
};
use crate::common::{CircuitTableAir, NpoAirBuilder, NpoPreprocessor};
use crate::config::StarkField;
use crate::{ConstraintProfile, impl_table_prover_batch_instances_from_base};

impl<SC, const D: usize> BatchAir<SC> for PermuteAir<Val<SC>, D>
where
    SC: StarkGenericConfig + Send + Sync,
    Val<SC>: StarkField,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
{
}

/// Table prover for the permute (multiset-equality) NPO.
///
/// `lanes` controls how many (input, output) pair-rows are packed per AIR row.
/// Pinned to 1 for the first cut; lane packing is a future optimisation.
pub struct PermuteProver<const D: usize> {
    lanes: usize,
}

impl<const D: usize> PermuteProver<D> {
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
        let op_type = NpoTypeId::permute();
        let trace = traces.non_primitive_traces.get(&op_type)?;
        if trace.rows() == 0 {
            return None;
        }

        let t = trace.as_any().downcast_ref::<PermuteTrace<Val<SC>>>()?;

        let num_pairs = t.total_rows();
        let lanes = packing.npo_lanes(&op_type).unwrap_or(self.lanes);
        let min_height = packing.min_trace_height();

        let prep_lane_width = PermuteAir::<Val<SC>, D>::preprocessed_lane_width();
        let mut preprocessed = Val::<SC>::zero_vec(num_pairs * prep_lane_width);
        let mut pair_idx = 0usize;
        for op in t.operations.iter() {
            for (&in_wid, &out_wid) in op.input_wids.iter().zip(op.output_wids.iter()) {
                let base = pair_idx * prep_lane_width;
                preprocessed[base + PERMUTE_PREP_IN_IDX] =
                    in_wid.base_field_index::<Val<SC>, D>();
                preprocessed[base + PERMUTE_PREP_IN_MULT] = Val::<SC>::ONE;
                preprocessed[base + PERMUTE_PREP_OUT_IDX] =
                    out_wid.base_field_index::<Val<SC>, D>();
                // out_mult placeholder — overwritten by PermutePreprocessor based on ext_reads.
                preprocessed[base + PERMUTE_PREP_OUT_MULT] = Val::<SC>::ONE;
                preprocessed[base + PERMUTE_PREP_IS_REAL] = Val::<SC>::ONE;
                pair_idx += 1;
            }
        }

        let air = PermuteAir::<Val<SC>, D>::new_with_preprocessed(lanes, preprocessed, min_height);
        let matrix = PermuteAir::<Val<SC>, D>::trace_to_matrix(&t.operations, lanes);

        Some(BatchTableInstance {
            op_type,
            air: DynamicAirEntry::new(Box::new(air)),
            trace: matrix,
            public_values: Vec::new(),
            rows: num_pairs,
            lanes,
        })
    }
}

impl<SC, const D: usize> TableProver<SC> for PermuteProver<D>
where
    SC: StarkGenericConfig + 'static + Send + Sync,
    Val<SC>: StarkField,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
{
    fn op_type(&self) -> NpoTypeId {
        NpoTypeId::permute()
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
        let air =
            PermuteAir::<Val<SC>, D>::new_with_preprocessed(table_entry.lanes, Vec::new(), 1);
        Ok(DynamicAirEntry::new(Box::new(air)))
    }

    fn air_with_committed_preprocessed(
        &self,
        committed_prep: Vec<Val<SC>>,
        min_height: usize,
        lanes: usize,
        _circuit_extension_degree: u32,
    ) -> Option<DynamicAirEntry<SC>> {
        let air = PermuteAir::<Val<SC>, D>::new_with_preprocessed(
            lanes,
            committed_prep,
            min_height,
        );
        Some(DynamicAirEntry::new(Box::new(air)))
    }
}

// ============================================================================
// Preprocessor — patches out_mult based on ext_reads, mirrors RecomposePreprocessor.
// ============================================================================

/// NpoPreprocessor for the permute table.
///
/// Converts EF preprocessed data to BF and patches `out_mult` (col 3) per row from
/// `ext_reads[output_wid]` (or `-1` for duplicate creators).
#[derive(Default, Clone)]
pub struct PermutePreprocessor;

impl PermutePreprocessor {
    pub const fn new() -> Self {
        Self
    }
}

impl NpoPreprocessor<KoalaBear> for PermutePreprocessor {
    fn preprocess(
        &self,
        _circuit: &dyn core::any::Any,
        preprocessed: &mut dyn core::any::Any,
    ) -> Result<NonPrimitivePreprocessedMap<KoalaBear>, CircuitError> {
        type F = KoalaBear;
        if let Some(prep) =
            preprocessed.downcast_mut::<PreprocessedColumns<BinomialExtensionField<F, 4>, 4>>()
        {
            return permute_preprocess_impl::<F, _, 4>(prep);
        }
        if let Some(prep) =
            preprocessed.downcast_mut::<PreprocessedColumns<QuinticTrinomialExtensionField<F>, 5>>()
        {
            return permute_preprocess_impl::<F, _, 5>(prep);
        }
        if let Some(prep) = preprocessed.downcast_mut::<PreprocessedColumns<F, 1>>() {
            return permute_preprocess_impl::<F, _, 1>(prep);
        }
        Ok(HashMap::new())
    }
}

impl NpoPreprocessor<BabyBear> for PermutePreprocessor {
    fn preprocess(
        &self,
        _circuit: &dyn core::any::Any,
        preprocessed: &mut dyn core::any::Any,
    ) -> Result<NonPrimitivePreprocessedMap<BabyBear>, CircuitError> {
        type F = BabyBear;
        if let Some(prep) =
            preprocessed.downcast_mut::<PreprocessedColumns<BinomialExtensionField<F, 4>, 4>>()
        {
            return permute_preprocess_impl::<F, _, 4>(prep);
        }
        if let Some(prep) = preprocessed.downcast_mut::<PreprocessedColumns<F, 1>>() {
            return permute_preprocess_impl::<F, _, 1>(prep);
        }
        Ok(HashMap::new())
    }
}

impl NpoPreprocessor<Goldilocks> for PermutePreprocessor {
    fn preprocess(
        &self,
        _circuit: &dyn core::any::Any,
        preprocessed: &mut dyn core::any::Any,
    ) -> Result<NonPrimitivePreprocessedMap<Goldilocks>, CircuitError> {
        type F = Goldilocks;
        if let Some(prep) =
            preprocessed.downcast_mut::<PreprocessedColumns<BinomialExtensionField<F, 2>, 2>>()
        {
            return permute_preprocess_impl::<F, _, 2>(prep);
        }
        if let Some(prep) = preprocessed.downcast_mut::<PreprocessedColumns<F, 1>>() {
            return permute_preprocess_impl::<F, _, 1>(prep);
        }
        Ok(HashMap::new())
    }
}

fn permute_preprocess_impl<F, EF, const D: usize>(
    prep: &PreprocessedColumns<EF, D>,
) -> Result<NonPrimitivePreprocessedMap<F>, CircuitError>
where
    F: StarkField + PrimeField64,
    EF: Field + ExtensionField<F> + 'static,
{
    let op_type = NpoTypeId::permute();
    let ef_data = match prep.non_primitive.get(&op_type) {
        Some(d) if !d.is_empty() => d,
        _ => return Ok(HashMap::new()),
    };

    let mut prep_base: Vec<F> = ef_data
        .iter()
        .map(|v| v.as_base().ok_or(CircuitError::InvalidPreprocessedValues))
        .collect::<Result<Vec<_>, CircuitError>>()?;

    if !prep_base.len().is_multiple_of(PERMUTE_PREP_LANE_WIDTH) {
        return Err(CircuitError::InvalidPreprocessedValues);
    }

    let neg_one = F::ZERO - F::ONE;
    let num_rows = prep_base.len() / PERMUTE_PREP_LANE_WIDTH;

    for row_idx in 0..num_rows {
        let row_start = row_idx * PERMUTE_PREP_LANE_WIDTH;

        // Patch out_mult based on ext_reads of the output_wid.
        let output_idx_val = prep_base[row_start + PERMUTE_PREP_OUT_IDX];
        let out_wid = F::as_canonical_u64(&output_idx_val) as usize / D;

        let is_dup = prep
            .dup_npo_outputs
            .get(&op_type)
            .and_then(|d| d.get(out_wid).copied())
            .unwrap_or(false);

        prep_base[row_start + PERMUTE_PREP_OUT_MULT] = if is_dup {
            neg_one
        } else {
            let n_reads = prep.ext_reads.get(out_wid).copied().unwrap_or(0);
            F::from_u32(n_reads)
        };
    }

    let mut result = HashMap::new();
    result.insert(op_type, prep_base);
    Ok(result)
}

// ============================================================================
// AIR Builder
// ============================================================================

/// NpoAirBuilder for the permute table.
#[derive(Clone)]
pub struct PermuteAirBuilder<const D: usize> {
    lanes: usize,
}

impl<const D: usize> PermuteAirBuilder<D> {
    pub fn new(lanes: usize) -> Self {
        Self {
            lanes: lanes.max(1),
        }
    }
}

impl<SC, const D: usize> NpoAirBuilder<SC, D> for PermuteAirBuilder<D>
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
        if op_type.as_str() != "permute" {
            return None;
        }

        let num_pair_rows = prep_base.len() / PERMUTE_PREP_LANE_WIDTH;
        let num_rows = num_pair_rows.div_ceil(lanes).max(1);

        let air = PermuteAir::<Val<SC>, D>::new_with_preprocessed(
            lanes,
            prep_base.to_vec(),
            min_height,
        );

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
