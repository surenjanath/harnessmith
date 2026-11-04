//! A bit-exact reimplementation of CPython's `random.Random` seeding and `shuffle`.
//!
//! The method's stream order (`schedule::stream_order`) and anonymous candidate ids
//! (`selection::assign_ids`) are defined by seeded Python shuffles; reproducing them exactly
//! keeps runs, caches and namespaces interchangeable with the reference implementation.

use sha2::{Digest, Sha512};

const N: usize = 624;
const M: usize = 397;

pub struct PyRandom {
    mt: [u32; N],
    mti: usize,
}

impl PyRandom {
    fn init_genrand(s: u32) -> PyRandom {
        let mut mt = [0u32; N];
        mt[0] = s;
        for i in 1..N {
            mt[i] = 1812433253u32
                .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        PyRandom { mt, mti: N }
    }

    /// `init_by_array` from CPython's `_randommodule.c`.
    fn init_by_array(key: &[u32]) -> PyRandom {
        let mut r = PyRandom::init_genrand(19650218);
        let mt = &mut r.mt;
        let (mut i, mut j) = (1usize, 0usize);
        let klen = key.len();
        let mut k = N.max(klen);
        while k > 0 {
            mt[i] = (mt[i] ^ (mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(1664525))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                mt[0] = mt[N - 1];
                i = 1;
            }
            if j >= klen {
                j = 0;
            }
            k -= 1;
        }
        k = N - 1;
        while k > 0 {
            mt[i] = (mt[i] ^ (mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(1566083941))
                .wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                mt[0] = mt[N - 1];
                i = 1;
            }
            k -= 1;
        }
        mt[0] = 0x8000_0000;
        r
    }

    /// Seed from big-endian magnitude bytes of a non-negative integer.
    fn from_be_bytes(be: &[u8]) -> PyRandom {
        let mut le: Vec<u8> = be.iter().rev().cloned().collect();
        while le.len() % 4 != 0 {
            le.push(0);
        }
        let mut key: Vec<u32> = le
            .chunks(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        while key.len() > 1 && *key.last().unwrap() == 0 {
            key.pop();
        }
        if key.is_empty() {
            key.push(0);
        }
        PyRandom::init_by_array(&key)
    }

    /// `random.Random(n)` for an integer seed (the absolute value is used, as in CPython).
    pub fn from_int(n: i128) -> PyRandom {
        let m = n.unsigned_abs();
        let be = m.to_be_bytes();
        let first = be.iter().position(|b| *b != 0).unwrap_or(be.len());
        PyRandom::from_be_bytes(&be[first..])
    }

    /// `random.Random(s)` for a str seed (version 2: `s + sha512(s)` as a big-endian int).
    pub fn from_str(s: &str) -> PyRandom {
        let mut bytes = s.as_bytes().to_vec();
        bytes.extend_from_slice(&Sha512::digest(s.as_bytes()));
        let first = bytes.iter().position(|b| *b != 0).unwrap_or(bytes.len());
        PyRandom::from_be_bytes(&bytes[first..])
    }

    pub fn genrand_u32(&mut self) -> u32 {
        const MATRIX_A: u32 = 0x9908_b0df;
        const UPPER: u32 = 0x8000_0000;
        const LOWER: u32 = 0x7fff_ffff;
        if self.mti >= N {
            let mt = &mut self.mt;
            for kk in 0..N {
                let y = (mt[kk] & UPPER) | (mt[(kk + 1) % N] & LOWER);
                let v = mt[(kk + M) % N] ^ (y >> 1) ^ if y & 1 != 0 { MATRIX_A } else { 0 };
                mt[kk] = v;
            }
            self.mti = 0;
        }
        let mut y = self.mt[self.mti];
        self.mti += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// `getrandbits(k)` for k <= 64.
    pub fn getrandbits(&mut self, k: u32) -> u64 {
        if k == 0 {
            return 0;
        }
        if k <= 32 {
            return (self.genrand_u32() >> (32 - k)) as u64;
        }
        // CPython fills 32-bit words from least significant upward.
        let lo = self.genrand_u32() as u64;
        let rem = k - 32;
        let hi = (self.genrand_u32() >> (32 - rem)) as u64;
        lo | (hi << 32)
    }

    fn randbelow(&mut self, n: u64) -> u64 {
        let k = 64 - n.leading_zeros();
        let mut r = self.getrandbits(k);
        while r >= n {
            r = self.getrandbits(k);
        }
        r
    }

    /// `random.shuffle(x)`.
    pub fn shuffle<T>(&mut self, x: &mut [T]) {
        if x.len() < 2 {
            return;
        }
        for i in (1..x.len()).rev() {
            let j = self.randbelow(i as u64 + 1) as usize;
            x.swap(i, j);
        }
    }

    /// `random.random()`.
    pub fn random(&mut self) -> f64 {
        let a = (self.genrand_u32() >> 5) as f64;
        let b = (self.genrand_u32() >> 6) as f64;
        (a * 67108864.0 + b) * (1.0 / 9007199254740992.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference values computed with CPython 3.12.
    #[test]
    fn int_seed_matches_cpython() {
        let mut r = PyRandom::from_int(42);
        assert!((r.random() - 0.6394267984578837).abs() < 1e-15);
        let mut v: Vec<i32> = (0..10).collect();
        PyRandom::from_int(42).shuffle(&mut v);
        assert_eq!(v, vec![7, 3, 2, 8, 5, 6, 9, 4, 0, 1]);
    }

    #[test]
    fn str_seed_matches_cpython() {
        let mut v: Vec<i32> = (0..10).collect();
        PyRandom::from_str("7:__domains__").shuffle(&mut v);
        assert_eq!(v, crate::core::pyrandom::tests::expected_str());
    }

    #[test]
    fn sha1_int_seed_matches_cpython() {
        let mut v: Vec<i32> = (0..5).collect();
        PyRandom::from_int(14539092345157).shuffle(&mut v);
        assert_eq!(v, vec![1, 3, 4, 2, 0]);
    }

    pub fn expected_str() -> Vec<i32> {
        // python3 -c "import random;v=list(range(10));random.Random('7:__domains__').shuffle(v);print(v)"
        vec![9, 1, 7, 5, 8, 2, 3, 6, 4, 0]
    }
}
