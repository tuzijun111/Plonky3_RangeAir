//! REVIEW probe: decomposing a select output (one branch with provenance) in a circuit that
//! never enabled the recompose NPO.
use p3_baby_bear::BabyBear;
use p3_circuit::CircuitBuilder;
use p3_field::extension::BinomialExtensionField;

type BF = BabyBear;
type EF = BinomialExtensionField<BF, 4>;

#[test]
fn select_decompose_without_recompose_npo() {
    let mut b = CircuitBuilder::<EF>::new();
    let t = b.public_input();
    let sb: Vec<_> = (0..4).map(|_| b.public_input()).collect();
    let s = b.recompose_base_coeffs_to_ext::<BF>(&sb).unwrap();
    let sel = b.public_input();
    let x = b.select(sel, t, s);
    let r = b.decompose_ext_to_base_coeffs::<BF>(x);
    println!("decompose(select) without recompose NPO: {r:?}");
    assert!(r.is_ok(), "{r:?}");
    b.build().expect("build");
}
