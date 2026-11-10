//! `calc`: exact arithmetic for models that should not do it in their heads.
//!
//! A small expression evaluator: numbers (with `1,234.5` thousands separators and currency
//! symbols), `+ - * / // % ^ **`, parentheses, percentages (`15%` is 0.15, so `80 * 15%` works),
//! and the usual functions. It runs in-process, so it works when the shell is turned off.

use anyhow::{bail, Result};

struct P<'a> {
    s: &'a [u8],
    i: usize,
    /// Depth of function-call parentheses: inside them a comma separates arguments.
    fdepth: usize,
}

const HELP: &str = "Use numbers, + - * / ^ ( ), a % sign for percentages, and functions: sqrt abs round floor ceil min max sum avg median pow log ln log10 exp";

impl<'a> P<'a> {
    fn ws(&mut self) {
        while self.i < self.s.len() && (self.s[self.i] as char).is_whitespace() {
            self.i += 1;
        }
    }
    fn peek(&mut self) -> Option<u8> {
        self.ws();
        self.s.get(self.i).copied()
    }
    fn eat(&mut self, c: u8) -> bool {
        if self.peek() == Some(c) {
            self.i += 1;
            true
        } else {
            false
        }
    }
    fn expr(&mut self) -> Result<f64> {
        let mut v = self.term()?;
        loop {
            if self.eat(b'+') {
                v += self.term()?;
            } else if self.eat(b'-') {
                v -= self.term()?;
            } else {
                return Ok(v);
            }
        }
    }
    fn term(&mut self) -> Result<f64> {
        let mut v = self.unary()?;
        loop {
            match self.peek() {
                Some(b'*') if self.s.get(self.i + 1) != Some(&b'*') => {
                    self.i += 1;
                    v *= self.unary()?;
                }
                Some(b'/') => {
                    self.i += 1;
                    let floor = self.s.get(self.i) == Some(&b'/');
                    if floor {
                        self.i += 1;
                    }
                    let d = self.unary()?;
                    if d == 0.0 {
                        bail!("division by zero");
                    }
                    v = if floor { (v / d).floor() } else { v / d };
                }
                Some(b'%') => {
                    self.i += 1;
                    let d = self.unary()?;
                    if d == 0.0 {
                        bail!("modulo by zero");
                    }
                    v = v.rem_euclid(d);
                }
                _ => return Ok(v),
            }
        }
    }
    fn unary(&mut self) -> Result<f64> {
        if self.eat(b'-') {
            return Ok(-self.unary()?);
        }
        if self.eat(b'+') {
            return self.unary();
        }
        self.power()
    }
    fn power(&mut self) -> Result<f64> {
        let base = self.postfix()?;
        self.ws();
        let pow = if self.s[self.i..].starts_with(b"**") {
            self.i += 2;
            true
        } else {
            self.eat(b'^')
        };
        if pow {
            let e = self.unary()?;
            return Ok(base.powf(e));
        }
        Ok(base)
    }
    /// A value, optionally followed by a percent sign (when no operand follows it).
    fn postfix(&mut self) -> Result<f64> {
        let v = self.primary()?;
        if self.peek() == Some(b'%') {
            let mut j = self.i + 1;
            while j < self.s.len() && (self.s[j] as char).is_whitespace() {
                j += 1;
            }
            let operand_follows = self.s.get(j).map(|c| c.is_ascii_alphanumeric() || *c == b'(' || *c == b'.').unwrap_or(false);
            if !operand_follows {
                self.i += 1;
                return Ok(v / 100.0);
            }
        }
        Ok(v)
    }
    fn primary(&mut self) -> Result<f64> {
        match self.peek() {
            None => bail!("the expression ends too early. {HELP}"),
            Some(b'(') => {
                self.i += 1;
                let saved = std::mem::replace(&mut self.fdepth, 0);
                let v = self.expr()?;
                self.fdepth = saved;
                if !self.eat(b')') {
                    bail!("missing a closing parenthesis");
                }
                Ok(v)
            }
            Some(c) if c.is_ascii_digit() || c == b'.' => self.number(),
            Some(c) if c.is_ascii_alphabetic() || c == b'_' => {
                let start = self.i;
                while self.i < self.s.len() && (self.s[self.i].is_ascii_alphanumeric() || self.s[self.i] == b'_') {
                    self.i += 1;
                }
                let name = String::from_utf8_lossy(&self.s[start..self.i]).to_lowercase();
                if self.eat(b'(') {
                    self.fdepth += 1;
                    let mut args = vec![];
                    if self.peek() != Some(b')') {
                        loop {
                            args.push(self.expr()?);
                            if !self.eat(b',') {
                                break;
                            }
                        }
                    }
                    self.fdepth -= 1;
                    if !self.eat(b')') {
                        bail!("missing a closing parenthesis after {name}(…");
                    }
                    return func(&name, &args);
                }
                match name.as_str() {
                    "pi" => Ok(std::f64::consts::PI),
                    "e" => Ok(std::f64::consts::E),
                    _ => bail!("unknown name `{name}`: calc has no variables, write the numbers themselves. {HELP}"),
                }
            }
            Some(c) => bail!("unexpected `{}` at position {}. {HELP}", c as char, self.i + 1),
        }
    }
    fn number(&mut self) -> Result<f64> {
        let mut t = String::new();
        let digits = |p: &mut Self, t: &mut String| {
            while p.i < p.s.len() && (p.s[p.i].is_ascii_digit() || p.s[p.i] == b'_') {
                if p.s[p.i] != b'_' {
                    t.push(p.s[p.i] as char);
                }
                p.i += 1;
            }
        };
        digits(self, &mut t);
        // 1,234,567: a comma followed by exactly three digits, outside function arguments
        while self.fdepth == 0 && self.s.get(self.i) == Some(&b',') && self.s.len() >= self.i + 4 && self.s[self.i + 1..self.i + 4].iter().all(|c| c.is_ascii_digit()) && !self.s.get(self.i + 4).map(|c| c.is_ascii_digit()).unwrap_or(false) {
            t.push_str(&String::from_utf8_lossy(&self.s[self.i + 1..self.i + 4]));
            self.i += 4;
        }
        if self.s.get(self.i) == Some(&b'.') && self.s.get(self.i + 1).map(|c| c.is_ascii_digit()).unwrap_or(false) {
            t.push('.');
            self.i += 1;
            digits(self, &mut t);
        } else if self.s.get(self.i) == Some(&b'.') && !t.is_empty() {
            self.i += 1;
        }
        if matches!(self.s.get(self.i), Some(b'e') | Some(b'E')) {
            let mut j = self.i + 1;
            if matches!(self.s.get(j), Some(b'+') | Some(b'-')) {
                j += 1;
            }
            if self.s.get(j).map(|c| c.is_ascii_digit()).unwrap_or(false) {
                t.push_str(&String::from_utf8_lossy(&self.s[self.i..j]));
                self.i = j;
                digits(self, &mut t);
            }
        }
        t.parse::<f64>().map_err(|_| anyhow::anyhow!("`{t}` is not a number"))
    }
}

fn func(name: &str, a: &[f64]) -> Result<f64> {
    let one = || -> Result<f64> {
        if a.len() == 1 { Ok(a[0]) } else { bail!("{name}() takes one value, got {}. Inside a function, write numbers without thousands separators (1480, not 1,480)", a.len()) }
    };
    let some = || -> Result<()> {
        if a.is_empty() { bail!("{name}() needs at least one value") } else { Ok(()) }
    };
    Ok(match name {
        "sqrt" => {
            let x = one()?;
            if x < 0.0 {
                bail!("sqrt of a negative number");
            }
            x.sqrt()
        }
        "abs" => one()?.abs(),
        "floor" | "int" | "trunc" => {
            let x = one()?;
            if name == "floor" { x.floor() } else { x.trunc() }
        }
        "ceil" => one()?.ceil(),
        "round" => match a {
            [x] => round_half_even(*x, 0),
            [x, n] => round_half_even(*x, *n as i32),
            _ => bail!("round(value) or round(value, decimals). Inside a function, write numbers without thousands separators (1480, not 1,480)"),
        },
        "min" => {
            some()?;
            a.iter().cloned().fold(f64::INFINITY, f64::min)
        }
        "max" => {
            some()?;
            a.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        }
        "sum" => a.iter().sum(),
        "avg" | "mean" | "average" => {
            some()?;
            a.iter().sum::<f64>() / a.len() as f64
        }
        "median" => {
            some()?;
            let mut v = a.to_vec();
            v.sort_by(|x, y| x.partial_cmp(y).unwrap());
            let n = v.len();
            if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
        }
        "pow" => match a {
            [x, y] => x.powf(*y),
            _ => bail!("pow(base, exponent)"),
        },
        "exp" => one()?.exp(),
        "ln" => one()?.ln(),
        "log10" => one()?.log10(),
        "log2" => one()?.log2(),
        "log" => match a {
            [x] => x.ln(),
            [x, b] => x.ln() / b.ln(),
            _ => bail!("log(value) or log(value, base)"),
        },
        "sin" => one()?.sin(),
        "cos" => one()?.cos(),
        "tan" => one()?.tan(),
        "fact" | "factorial" => {
            let x = one()?;
            if x < 0.0 || x.fract() != 0.0 || x > 170.0 {
                bail!("factorial needs a whole number from 0 to 170");
            }
            (1..=x as u64).map(|k| k as f64).product()
        }
        _ => bail!("unknown function `{name}`. {HELP}"),
    })
}

/// Python-style rounding (ties to even on the decimal representation is not attempted: ties
/// away from zero on the scaled value, which is what people expect from a calculator).
fn round_half_even(x: f64, places: i32) -> f64 {
    let m = 10f64.powi(places.clamp(-12, 12));
    // The tiny nudge undoes binary noise such as 1352.345 being stored as 1352.3449999.
    ((x * m) + (x * m).signum() * 1e-9).round() / m
}

/// Evaluate an arithmetic expression.
pub fn eval(expr: &str) -> Result<f64> {
    let cleaned: String = expr
        .replace('×', "*")
        .replace('÷', "/")
        .replace('−', "-")
        .replace('–', "-")
        .chars()
        .filter(|c| !matches!(c, '$' | '€' | '£' | '¥' | '=' | '?'))
        .collect();
    let cleaned = cleaned.trim();
    if cleaned.is_empty() {
        bail!("empty expression. {HELP}");
    }
    let mut p = P { s: cleaned.as_bytes(), i: 0, fdepth: 0 };
    let v = p.expr()?;
    if p.peek().is_some() {
        let rest = &cleaned[p.i..];
        bail!("could not read `{}` (position {}). {HELP}", crate::util::take_chars(rest, 20), p.i + 1);
    }
    if !v.is_finite() {
        bail!("the result is not a finite number");
    }
    Ok(v)
}

/// A number as a person would write it: no float noise, no trailing zeros.
pub fn format(v: f64) -> String {
    if v == v.trunc() && v.abs() < 1e15 {
        return format!("{}", v as i64);
    }
    let s = format!("{:.10}", v);
    let s = s.trim_end_matches('0').trim_end_matches('.').to_string();
    if s == "-0" { "0".into() } else { s }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(s: &str) -> String {
        format(eval(s).unwrap())
    }

    #[test]
    fn arithmetic_percentages_and_functions() {
        assert_eq!(ev("120*4.50 + 75*12"), "1440");
        assert_eq!(ev("1,250 - 387 - 462 + 3 * 144"), "833");
        assert_eq!(ev("round($1480 * (1 - 15%) * (1 + 7.5%), 2)"), "1352.35");
        assert_eq!(ev("$1,480 * (1 - 15%)"), "1258");
        assert!(eval("round(1,480 * 0.85, 2)").unwrap_err().to_string().contains("without thousands separators"));
        assert_eq!(ev("2400 * 1.035 ^ 6"), ev("2400 * 1.035 ** 6"));
        assert_eq!(ev("round(2400 * 1.035 ** 6, 2)"), "2950.21");
        assert_eq!(ev("17 % 5"), "2");
        assert_eq!(ev("17 // 5"), "3");
        assert_eq!(ev("-2 ^ 2"), "-4");
        assert_eq!(ev("max(1,234, 99)"), "234");
        assert_eq!(ev("median(47, 12, 88, 35)"), "41");
        assert_eq!(ev("avg(1, 2, 3, 4)"), "2.5");
        assert_eq!(ev("round(135000 / 6480, 2)"), "20.83");
        assert_eq!(ev("sqrt(2) * sqrt(2)"), "2");
        assert_eq!(ev("0.1 + 0.2"), "0.3");
        assert_eq!(ev("50% of 3".replace(" of ", " * ").as_str()), "1.5");
    }

    #[test]
    fn errors_say_what_to_do() {
        assert!(eval("total * 2").unwrap_err().to_string().contains("no variables"));
        assert!(eval("3 / 0").unwrap_err().to_string().contains("division by zero"));
        assert!(eval("(3 + 4").unwrap_err().to_string().contains("closing parenthesis"));
        assert!(eval("3 + ").unwrap_err().to_string().contains("ends too early"));
        assert!(eval("3 4").is_err());
    }
}
