//! Optional login for the UI. Set HARNESS2_UI_PASSWORD (8+ characters) and every page and API
//! call needs either the session cookie from /login or `Authorization: Bearer <password>`.
//! Inbound hooks keep their own tokens and are not behind the login.

use axum::extract::Request;
use axum::http::{header, HeaderMap, StatusCode};
use axum::middleware::Next;
use axum::response::{Html, IntoResponse, Redirect, Response};
use axum::Form;
use once_cell::sync::Lazy;
use serde::Deserialize;

const COOKIE: &str = "hsm_session";

pub fn password() -> Option<String> {
    crate::config::env("HARNESS2_UI_PASSWORD").filter(|p| p.chars().count() >= 8)
}

/// A per-process secret, so sessions end when the server restarts.
static SECRET: Lazy<Vec<u8>> = Lazy::new(|| {
    let mut b = vec![0u8; 32];
    if let Ok(mut f) = std::fs::File::open("/dev/urandom") {
        use std::io::Read;
        let _ = f.read_exact(&mut b);
    }
    if b.iter().all(|x| *x == 0) {
        b = crate::util::sha256_hex(format!("{:?}{}", std::time::SystemTime::now(), std::process::id()).as_bytes()).into_bytes();
    }
    b
});

fn session_value(pw: &str) -> String {
    super::inbound::hmac_sha256_hex(&SECRET, pw.as_bytes())
}

fn ct_eq(a: &str, b: &str) -> bool {
    a.len() == b.len() && a.bytes().zip(b.bytes()).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn authed(h: &HeaderMap, pw: &str) -> bool {
    let want = session_value(pw);
    let cookie_ok = h
        .get_all(header::COOKIE)
        .iter()
        .filter_map(|v| v.to_str().ok())
        .flat_map(|v| v.split(';'))
        .filter_map(|kv| kv.trim().split_once('='))
        .any(|(k, v)| k == COOKIE && ct_eq(v, &want));
    let bearer_ok = h.get(header::AUTHORIZATION).and_then(|v| v.to_str().ok()).and_then(|v| v.strip_prefix("Bearer ")).map(|t| ct_eq(t, pw)).unwrap_or(false);
    cookie_ok || bearer_ok
}

pub async fn guard(req: Request, next: Next) -> Response {
    let Some(pw) = password() else { return next.run(req).await };
    let path = req.uri().path().to_string();
    if path == "/login" || path == "/logout" || path.starts_with("/api/hooks/") || authed(req.headers(), &pw) {
        return next.run(req).await;
    }
    if path.starts_with("/api/") {
        return (StatusCode::UNAUTHORIZED, axum::Json(serde_json::json!({"error": "sign in first"}))).into_response();
    }
    Redirect::to("/login").into_response()
}

pub async fn login_page() -> Html<String> {
    Html(page(""))
}

#[derive(Deserialize)]
pub struct LoginForm {
    password: String,
}

pub async fn login_post(Form(f): Form<LoginForm>) -> Response {
    let Some(pw) = password() else { return Redirect::to("/").into_response() };
    if !ct_eq(&f.password, &pw) {
        tokio::time::sleep(std::time::Duration::from_millis(800)).await;
        return (StatusCode::UNAUTHORIZED, Html(page("That password is not right."))).into_response();
    }
    let cookie = format!("{COOKIE}={}; HttpOnly; SameSite=Strict; Path=/; Max-Age=2592000", session_value(&pw));
    ([(header::SET_COOKIE, cookie)], Redirect::to("/")).into_response()
}

pub async fn logout() -> Response {
    ([(header::SET_COOKIE, format!("{COOKIE}=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0"))], Redirect::to("/login")).into_response()
}

fn page(err: &str) -> String {
    format!(
        r#"<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Harnessmith · sign in</title>
<style>
:root {{ --bg:#f5f6f8; --panel:#fff; --ink:#16191d; --muted:#5b6470; --line:#e2e5e9; --accent:#2563eb; --bad:#cc3b3b; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#000; --panel:#0b0b0c; --ink:#ededee; --muted:#9a9aa0; --line:#222225; --accent:#4f8ff0; --bad:#ec666d; }} }}
* {{ box-sizing: border-box; }}
body {{ margin:0; min-height:100vh; display:grid; place-items:center; background:var(--bg); color:var(--ink); font:14px/1.5 Inter, system-ui, sans-serif; padding:16px; }}
form {{ width:100%; max-width:340px; background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:24px; }}
h1 {{ font:700 22px Georgia, serif; margin:0 0 4px; }} p {{ color:var(--muted); margin:0 0 16px; font-size:13px; }}
input {{ width:100%; font:inherit; padding:9px 10px; border-radius:7px; border:1px solid var(--line); background:var(--bg); color:var(--ink); }}
button {{ margin-top:12px; width:100%; padding:9px; border:0; border-radius:7px; background:var(--accent); color:#fff; font:600 14px inherit; cursor:pointer; }}
.err {{ color:var(--bad); margin-top:10px; font-size:13px; }}
</style></head><body><form method="post" action="/login"><h1>Harnessmith</h1><p>This server asks for a password.</p>
<input type="password" name="password" autofocus autocomplete="current-password" placeholder="Password"><button type="submit">Sign in</button>{}</form></body></html>"#,
        if err.is_empty() { String::new() } else { format!(r#"<div class="err">{err}</div>"#) }
    )
}
