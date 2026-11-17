//! Numeric primitives. The scalar type and vector width live here so a later quantization
//! (f16 / i8) is a change to this file, not to the kernel.

/// Scalar type of every state variable.
pub type S = f32;
/// Width of every cell vector.
pub const D: usize = 32;
pub type V = [S; D];
/// Row-major D x D matrix: `out[i] = dot(m[i], x)`.
pub type M = [[S; D]; D];
pub const ZERO: V = [0.0; D];
pub const ZERO_M: M = [[0.0; D]; D];

const LANES: usize = 8;

#[inline]
pub fn dot(a: &V, b: &V) -> S {
    // Fixed 8-lane partial sums: vectorizes without fast-math and fixes the summation order (determinism).
    let mut acc = [0.0; LANES];
    for (ca, cb) in a
        .as_chunks::<LANES>()
        .0
        .iter()
        .zip(b.as_chunks::<LANES>().0)
    {
        for ((s, x), y) in acc.iter_mut().zip(ca).zip(cb) {
            *s += x * y;
        }
    }
    acc.iter().sum()
}

#[inline]
pub fn norm2(a: &V) -> S {
    dot(a, a)
}

#[inline]
pub fn norm(a: &V) -> S {
    norm2(a).sqrt()
}

#[inline]
pub fn matvec(m: &M, x: &V) -> V {
    std::array::from_fn(|i| dot(&m[i], x))
}

/// `m += s * a b^T`
#[inline]
pub fn outer_add(m: &mut M, a: &V, b: &V, s: S) {
    for (row, &ai) in m.iter_mut().zip(a) {
        axpy(row, s * ai, b);
    }
}

/// `y += a * x`
#[inline]
pub fn axpy(y: &mut V, a: S, x: &V) {
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi += a * xi;
    }
}

#[inline]
pub fn scale(x: &mut V, s: S) {
    for xi in x.iter_mut() {
        *xi *= s;
    }
}

#[inline]
pub fn sub(a: &V, b: &V) -> V {
    std::array::from_fn(|i| a[i] - b[i])
}

#[inline]
pub fn neg(a: &V) -> V {
    a.map(|x| -x)
}

/// Exact-range i8 code for vectors in [-1, 1] (cell states are tanh outputs).
pub fn quantize_unit(x: &V) -> [i8; D] {
    x.map(|v| (v.clamp(-1.0, 1.0) * 127.0).round() as i8)
}

#[cfg(test)]
pub(crate) fn basis(i: usize) -> V {
    let mut v = ZERO;
    v[i] = 1.0;
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(k: S) -> V {
        std::array::from_fn(|i| (i as S - 15.5) * k)
    }

    #[test]
    fn dot_matches_naive() {
        let (a, b) = (ramp(0.1), ramp(-0.3));
        let naive: f64 = a.iter().zip(&b).map(|(x, y)| *x as f64 * *y as f64).sum();
        assert!((dot(&a, &b) as f64 - naive).abs() < 1e-4);
    }

    #[test]
    fn matvec_and_outer() {
        let mut m = ZERO_M;
        outer_add(&mut m, &basis(1), &basis(2), 3.0);
        assert_eq!(m[1][2], 3.0);
        let mut want = ZERO;
        want[1] = 3.0;
        assert_eq!(matvec(&m, &basis(2)), want);
    }

    #[test]
    fn quantize_is_exact_at_ends() {
        let mut v = ZERO;
        v[0] = 1.0;
        v[1] = -1.0;
        v[2] = 0.5;
        let q = quantize_unit(&v);
        assert_eq!((q[0], q[1], q[2], q[3]), (127, -127, 64, 0));
    }
}
