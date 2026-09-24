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
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
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

/// `x` (public) is decomposed and only coefficient 2 is used, once, by an ALU op:
/// `c_2 + 1 == y` (public).
fn single_use_circuit(bind: bool) -> (Circuit<EF>, WitnessId) {
    let mut b = CircuitBuilder::<EF>::new();
    b.enable_recompose::<BF>(generate_recompose_trace::<BF, EF>);
    b.set_bind_recompose_coeffs(bind);
    let x = b.public_input();
    let y = b.public_input();
    let one = b.define_const(EF::ONE);
    let c = b.decompose_ext_to_base_coeffs::<BF>(x).unwrap();
    let s = b.add(c[2], one);
    b.connect(s, y);
    let circuit = b.build().unwrap();
    let x_wid = circuit.public_rows[0];
    (circuit, x_wid)
}

#[test]
fn honest_single_use_decomposition_proves() {
    for bind in [true, false] {
        let (circuit, _) = single_use_circuit(bind);
        prove(circuit, &[ext([3, 5, 7, 9]), EF::from_u64(8)], |_| {})
            .unwrap_or_else(|e| panic!("bind={bind}: {e}"));
    }
}

/// The single-use case: one ALU read of a decomposed coefficient. Without arbitration a
/// `recompose/coeff` row and the ALU's first use would both claim the creator role with
/// multiplicity 0, and the forged coefficient would verify.
#[test]
fn forged_single_use_coefficient_rejected() {
    let publics = [ext([3, 5, 7, 9]), EF::from_u64(1001)];
    let (circuit, x) = single_use_circuit(false);
    assert!(
        prove(circuit, &publics, |c| forge_decomposition(c, x, [3, 5, 1000, 9])).is_ok(),
        "PoC premise: the unbound lowering accepts a forged coefficient"
    );
    let (circuit, x) = single_use_circuit(true);
    let res = prove(circuit, &publics, |c| forge_decomposition(c, x, [3, 5, 1000, 9]));
    assert!(res.is_err(), "bound lowering must reject a forged coefficient");
}

/// A coefficient with an extension-valued forgery that keeps `sum c_i X^i == x` (what an ALU
/// reconnect alone would allow): `c_0 += X, c_1 -= 1`.
#[test]
fn forged_extension_valued_coefficient_rejected() {
    let mut b = CircuitBuilder::<EF>::new();
    b.enable_recompose::<BF>(generate_recompose_trace::<BF, EF>);
    let x = b.public_input();
    let y = b.public_input();
    let c = b.decompose_ext_to_base_coeffs::<BF>(x).unwrap();
    // Two ALU consumers so both forged coefficients are read.
    let s = b.add(c[0], c[1]);
    b.connect(s, y);
    let circuit = b.build().unwrap();
    let x_wid = circuit.public_rows[0];
    let honest = ext([3, 5, 7, 9]);
    // c_0' = 3 + X, c_1' = 5 - 1 = 4: c_0' + c_1'·X = 3 + 5X, unchanged; c_0' + c_1' = 7 + X.
    let c0 = ext([3, 1, 0, 0]);
    let c1 = EF::from_u64(4);
    let res = prove(circuit, &[honest, c0 + c1], |circ| {
        for op in &mut circ.ops {
            if let Op::Hint {
                inputs, executor, ..
            } = op
                && inputs.as_slice() == [x_wid]
            {
                *executor = Box::new(FixedHint(vec![c0, c1, EF::from_u64(7), EF::from_u64(9)]));
            }
        }
    });
    assert!(res.is_err(), "an extension-valued coefficient must be rejected");
}

/// Two decompositions whose coefficients are all consumed by several ALU ops, plus a
/// coefficient pinned to a constant and a coefficient reused as the input of a later
/// recomposition (the challenger's unpack-then-repack shape).
fn shared_circuit(bind: bool) -> (Circuit<EF>, WitnessId) {
    let mut b = CircuitBuilder::<EF>::new();
    b.enable_recompose::<BF>(generate_recompose_trace::<BF, EF>);
    b.set_bind_recompose_coeffs(bind);
    let x = b.public_input();
    let z = b.public_input();
    let out = b.public_input();
    let zero = b.define_const(EF::ZERO);
    let cx = b.decompose_ext_to_base_coeffs::<BF>(x).unwrap();
    let cz = b.decompose_ext_to_base_coeffs::<BF>(z).unwrap();
    b.connect(cz[3], zero);
    let mut acc = b.mul(cx[0], cz[0]);
    for i in 1..D {
        let t = b.mul(cx[i], cx[i]);
        acc = b.add(acc, t);
    }
    // Repack coefficients of x and z into a new limb and fold it into the output.
    let r = b
        .recompose_base_coeffs_to_ext::<BF>(&[cx[1], cz[1], cx[3], cz[2]])
        .unwrap();
    let res = b.add(acc, r);
    b.connect(res, out);
    let circuit = b.build().unwrap();
    let x_wid = circuit.public_rows[0];
    (circuit, x_wid)
}

fn shared_expected(x: [u64; 4], z: [u64; 4]) -> EF {
    let f = |v: u64| EF::from_u64(v);
    let acc = f(x[0]) * f(z[0]) + f(x[1]) * f(x[1]) + f(x[2]) * f(x[2]) + f(x[3]) * f(x[3]);
    acc + ext([x[1], z[1], x[3], z[2]])
}

#[test]
fn honest_shared_and_repacked_coefficients_prove() {
    let (x, z) = ([3, 5, 7, 9], [11, 13, 17, 0]);
    for bind in [true, false] {
        let (circuit, _) = shared_circuit(bind);
        prove(circuit, &[ext(x), ext(z), shared_expected(x, z)], |_| {})
            .unwrap_or_else(|e| panic!("bind={bind}: {e}"));
    }
}

#[test]
fn forged_shared_coefficient_rejected() {
    let (x, z) = ([3, 5, 7, 9], [11, 13, 17, 0]);
    let forged = [3, 6, 7, 9];
    // The forged run's arithmetic and repacking follow the forged coefficients.
    let publics = [ext(x), ext(z), shared_expected(forged, z)];
    let (circuit, xw) = shared_circuit(false);
    assert!(
        prove(circuit, &publics, |c| forge_decomposition(c, xw, forged)).is_ok(),
        "PoC premise: the unbound lowering accepts a forged coefficient"
    );
    let (circuit, xw) = shared_circuit(true);
    assert!(
        prove(circuit, &publics, |c| forge_decomposition(c, xw, forged)).is_err(),
        "bound lowering must reject a forged coefficient"
    );
}

/// An explicit recomposition of four public base values whose output feeds an ALU op.
fn explicit_recompose_circuit(bind: bool) -> (Circuit<EF>, Vec<WitnessId>) {
    let mut b = CircuitBuilder::<EF>::new();
    b.enable_recompose::<BF>(generate_recompose_trace::<BF, EF>);
    b.set_bind_recompose_coeffs(bind);
    let a: Vec<_> = (0..D).map(|_| b.public_input()).collect();
    let q = b.public_input();
    let r = b.recompose_base_coeffs_to_ext::<BF>(&a).unwrap();
    let sq = b.mul(r, r);
    b.connect(sq, q);
    let circuit = b.build().unwrap();
    let wids = circuit.public_rows[..D].to_vec();
    (circuit, wids)
}

#[test]
fn honest_explicit_recompose_proves() {
    let r = ext([2, 3, 4, 5]);
    let publics = [2, 3, 4, 5].map(EF::from_u64);
    for bind in [true, false] {
        let (circuit, _) = explicit_recompose_circuit(bind);
        prove(circuit, &[publics.as_slice(), &[r * r]].concat(), |_| {})
            .unwrap_or_else(|e| panic!("bind={bind}: {e}"));
    }
}

/// The recomposition's output is replaced by a value unrelated to its inputs.
#[test]
fn forged_explicit_recompose_output_rejected() {
    let fake = ext([100, 0, 0, 1]);
    let publics = [[2, 3, 4, 5].map(EF::from_u64).as_slice(), &[fake * fake]].concat();
    let (circuit, a) = explicit_recompose_circuit(false);
    assert!(
        prove(circuit, &publics, |c| forge_recompose(c, &a, Some(fake))).is_ok(),
        "PoC premise: the unbound lowering accepts a forged recomposition"
    );
    let (circuit, a) = explicit_recompose_circuit(true);
    assert!(
        prove(circuit, &publics, |c| forge_recompose(c, &a, Some(fake))).is_err(),
        "bound lowering must reject a forged recomposition"
    );
}
