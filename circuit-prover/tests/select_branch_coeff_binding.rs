//! Soundness of the recompose NPO's coefficient binding at D = 4.
//!
//! A malicious prover may fill the witness however it likes; only the AIRs and their buses
//! constrain it. These tests build a circuit, compute its verifier-side data, and only then
//! swap the executors that fill the witness (the extension-decomposition hint and the recompose
//! op), so the resulting proof is exactly what such a prover could submit.
//!
//! With `CircuitBuilder::set_bind_recompose_coeffs(false)` (the plain `recompose` table) every
//! forgery below is ACCEPTED: the row publishes only `[output_idx, v_0..v_3]` from free columns,
//! so neither a decomposition's coefficient witnesses nor an explicit recomposition's inputs are
//! tied to anything. With the default (`recompose/coeff` + first-occurrence creator
//! arbitration) each one is rejected, and the honest shapes still prove.

use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind};

use p3_circuit::ops::recompose::{RecomposeCircuitRow, RecomposeExecutionState};
use p3_circuit::ops::{
    ExecutionContext, HintExecutor, NonPrimitiveExecutor, PreprocessedWriter,
    generate_recompose_trace,
};
use p3_circuit::{Circuit, CircuitBuilder, CircuitError, NpoTypeId, Op, WitnessId};
use p3_circuit_prover::batch_stark_prover::{
    BatchStarkProver, CircuitProverData, TablePacking, recompose_air_builders,
    recompose_preprocessor,
};
use p3_circuit_prover::common::get_airs_and_degrees_with_prep;
use p3_circuit_prover::{ConstraintProfile, config};
use p3_field::extension::BinomialExtensionField;
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing, PrimeField64};
use p3_test_utils::baby_bear_params::*;

type BF = BabyBear;
type EF = BinomialExtensionField<BabyBear, 4>;
const D: usize = 4;

fn ext(c: [u64; 4]) -> EF {
    EF::from_basis_coefficients_fn(|i| BF::from_u64(c[i]))
}

fn embed(v: BF) -> EF {
    EF::from_basis_coefficients_fn(|i| if i == 0 { v } else { BF::ZERO })
}

/// Hint executor writing fixed values (a forged decomposition).
#[derive(Debug, Clone)]
struct FixedHint(Vec<EF>);

impl HintExecutor<EF> for FixedHint {
    fn execute(
        &self,
        _inputs: &[WitnessId],
        outputs: &[WitnessId],
        witness: &mut [Option<EF>],
    ) -> Result<(), CircuitError> {
        for (w, &v) in outputs.iter().zip(&self.0) {
            witness[w.0 as usize] = Some(v);
        }
        Ok(())
    }

    fn boxed(&self) -> Box<dyn HintExecutor<EF>> {
        Box::new(self.clone())
    }
}

/// Recompose executor that writes `out` to its output (instead of recomposing its inputs) and
/// records `out`'s true coefficients as the table row.
#[derive(Debug)]
struct ForcedRecompose {
    inner: Box<dyn NonPrimitiveExecutor<EF>>,
    /// `None`: copy the current value of the op's output witness (it is already set when the
    /// output is connected to an earlier-defined value, as in a decomposition).
    out: Option<EF>,
}

impl NonPrimitiveExecutor<EF> for ForcedRecompose {
    fn execute(
        &self,
        inputs: &[Vec<WitnessId>],
        outputs: &[Vec<WitnessId>],
        ctx: &mut ExecutionContext<'_, EF>,
    ) -> Result<(), CircuitError> {
        let out_wid = outputs[0][0];
        let v = match self.out {
            Some(v) => {
                ctx.set_witness(out_wid, v)?;
                v
            }
            None => ctx.get_witness(out_wid)?,
        };
        let values = v
            .as_basis_coefficients_slice()
            .iter()
            .map(|&c| embed(c))
            .collect();
        let op_type = self.inner.op_type().clone();
        ctx.get_op_state_mut::<RecomposeExecutionState<EF>>(&op_type)
            .rows
            .push(RecomposeCircuitRow {
                input_wids: inputs[0].clone(),
                output_wid: out_wid,
                values,
            });
        Ok(())
    }

    fn op_type(&self) -> &NpoTypeId {
        self.inner.op_type()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn preprocess(
        &self,
        inputs: &[Vec<WitnessId>],
        outputs: &[Vec<WitnessId>],
        preprocessed: &mut dyn PreprocessedWriter<EF>,
    ) -> Result<(), CircuitError> {
        self.inner.preprocess(inputs, outputs, preprocessed)
    }

    fn num_exposed_outputs(&self) -> Option<usize> {
        self.inner.num_exposed_outputs()
    }

    fn arbitrated_coeff_input_group(&self) -> Option<usize> {
        self.inner.arbitrated_coeff_input_group()
    }

    fn boxed(&self) -> Box<dyn NonPrimitiveExecutor<EF>> {
        Box::new(Self {
            inner: self.inner.boxed(),
            out: self.out,
        })
    }
}

fn is_recompose(op_type: &NpoTypeId) -> bool {
    *op_type == NpoTypeId::recompose() || *op_type == NpoTypeId::recompose_with_coeff_lookups()
}

/// Make the decomposition of witness `x` hand back `coeffs`, with the recompose row that
/// re-links them to `x` still recording `x`'s true coefficients.
fn forge_decomposition(circuit: &mut Circuit<EF>, x: WitnessId, coeffs: [u64; 4]) {
    let mut coeff_wids = None;
    for op in &mut circuit.ops {
        if let Op::Hint {
            inputs,
            outputs,
            executor,
        } = op
            && inputs.as_slice() == [x]
            && outputs.len() == D
        {
            *executor = Box::new(FixedHint(
                coeffs.iter().map(|&c| EF::from_u64(c)).collect(),
            ));
            coeff_wids = Some(outputs.clone());
        }
    }
    let coeff_wids = coeff_wids.expect("no decomposition hint for x");
    forge_recompose(circuit, &coeff_wids, None);
}

/// Replace the recompose op over `coeff_wids` by a [`ForcedRecompose`].
fn forge_recompose(circuit: &mut Circuit<EF>, coeff_wids: &[WitnessId], out: Option<EF>) {
    for op in &mut circuit.ops {
        if let Op::NonPrimitiveOpWithExecutor {
            inputs, executor, ..
        } = op
            && is_recompose(executor.op_type())
            && inputs[0] == coeff_wids
        {
            let inner = executor.boxed();
            *executor = Box::new(ForcedRecompose { inner, out });
            return;
        }
    }
    panic!("no recompose op over the given coefficients");
}

/// Compute the verifier data from the (honest) circuit, apply `tamper`, then prove and verify.
/// `Ok` iff the resulting proof verifies. The prover registers its recompose tables with
/// `split_coeff_tables = false`, as every pre-existing caller does.
fn prove(
    mut circuit: Circuit<EF>,
    publics: &[EF],
    tamper: impl FnOnce(&mut Circuit<EF>),
) -> Result<(), String> {
    let cfg = config::baby_bear();
    let (airs_degrees, primitive_columns, non_primitive_columns) =
        get_airs_and_degrees_with_prep::<config::BabyBearConfig, _, D>(
            &circuit,
            &TablePacking::default(),
            &[recompose_preprocessor::<BF>(false)],
            &recompose_air_builders::<config::BabyBearConfig, D>(1, false),
            ConstraintProfile::Standard,
        )
        .map_err(|e| format!("preprocess: {e:?}"))?;
    let (airs, log_degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();
    tamper(&mut circuit);

    let res = catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
        let mut runner = circuit.runner();
        runner
            .set_public_inputs(publics)
            .map_err(|e| format!("publics: {e:?}"))?;
        let traces = runner.run().map_err(|e| format!("run: {e:?}"))?;
        let prover_data =
            p3_batch_stark::ProverData::from_airs_and_degrees(&cfg, &airs, &log_degrees);
        let data = CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);
        let mut prover = BatchStarkProver::new(cfg);
        prover.register_recompose_table::<D>(false);
        let proof = prover
            .prove_all_tables(&traces, &data)
            .map_err(|e| format!("prove: {e:?}"))?;
        prover
            .verify_all_tables(&proof)
            .map_err(|e| format!("verify: {e:?}"))
    }));
    res.unwrap_or_else(|_| Err("panicked".into()))
}


/// REVIEW PoC: `x = select(sel, t, s)` where `s` has recompose provenance and `t` does not.
/// Decomposing `x` (bind mode, the default) decomposes `t` through the ALU chain and caches
/// those coefficients as `t`'s provenance; a later `decompose_ext_to_base_coeffs(t)` returns
/// them. `c_1 + 1 == y` (public).
fn select_branch_circuit() -> (Circuit<EF>, WitnessId) {
    let mut b = CircuitBuilder::<EF>::new();
    b.enable_recompose::<BF>(generate_recompose_trace::<BF, EF>);
    assert!(b.binds_recompose_coeffs());
    let t = b.public_input();
    let sb: Vec<_> = (0..D).map(|_| b.public_input()).collect();
    let sel = b.public_input();
    let y = b.public_input();
    let s = b.recompose_base_coeffs_to_ext::<BF>(&sb).unwrap();
    let x = b.select(sel, t, s);
    let cx = b.decompose_ext_to_base_coeffs::<BF>(x).unwrap();
    for &c in &cx {
        let _ = b.mul(c, c);
    }
    let ct = b.decompose_ext_to_base_coeffs::<BF>(t).unwrap();
    let one = b.define_const(EF::ONE);
    let r = b.add(ct[1], one);
    b.connect(r, y);
    let circuit = b.build().unwrap();
    let t_wid = circuit.public_rows[0];
    (circuit, t_wid)
}

fn select_publics(y: u64) -> Vec<EF> {
    let mut p = vec![ext([3, 5, 7, 9])];
    p.extend([11u64, 13, 17, 19].map(EF::from_u64));
    p.push(EF::ZERO); // sel = 0: x = s
    p.push(EF::from_u64(y));
    p
}

#[test]
fn select_branch_honest_proves() {
    let (circuit, _) = select_branch_circuit();
    prove(circuit, &select_publics(6), |_| {}).unwrap();
}

/// c_1' = c_1 + 10 (a BASE value, what a consumer of `decompose(t)[1]` reads) compensated by
/// the non-base c_0' = c_0 - 10·X: `sum c_i X^i == t` still holds, so the ALU chain accepts.
#[test]
fn select_branch_forged_base_coefficient() {
    let (circuit, t_wid) = select_branch_circuit();
    let c0 = ext([3, BF::ORDER_U64 - 10, 0, 0]);
    let res = prove(circuit, &select_publics(16), |circ| {
        let mut n = 0;
        for op in &mut circ.ops {
            if let Op::Hint { inputs, outputs, executor } = op
                && inputs.as_slice() == [t_wid]
                && outputs.len() == D
            {
                *executor = Box::new(FixedHint(vec![
                    c0,
                    EF::from_u64(15),
                    EF::from_u64(7),
                    EF::from_u64(9),
                ]));
                n += 1;
            }
        }
        assert_eq!(n, 1, "exactly one decomposition hint for t");
        // If the reconnect is a recompose NPO row (coeffbind-on-constfix), make it record t's
        // TRUE coefficients, as a malicious prover writing the trace directly would.
        let hint_out = circ.ops.iter().find_map(|op| match op {
            Op::Hint { inputs, outputs, .. } if inputs.as_slice() == [t_wid] => Some(outputs.clone()),
            _ => None,
        }).unwrap();
        let has_row = circ.ops.iter().any(|op| matches!(op, Op::NonPrimitiveOpWithExecutor { inputs, executor, .. } if is_recompose(executor.op_type()) && inputs[0] == hint_out));
        println!("reconnect through a recompose NPO row: {has_row}");
        if has_row {
            forge_recompose(circ, &hint_out, None);
        }
    });
    println!("forged select-branch coefficient: {res:?}");
    assert!(res.is_err(), "ACCEPTED: forged base coefficient of a select branch verifies");
}
