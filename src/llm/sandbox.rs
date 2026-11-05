//! Sandbox for the solver's shell commands (bash, wired tools, hooks).
//!
//! The gate refuses known-dangerous commands; the sandbox limits what any command can do:
//! the filesystem is read-only except the workspace, the home directory is hidden, task inputs
//! stay read-only, and the network is cut off unless the project allows it.
//!
//! Backends, best first: bubblewrap (`bwrap`), then `unshare` (user + mount + net namespaces).
//! "auto" picks the first that works on this machine; when neither does, commands run unsandboxed
//! and the Doctor reports it.

use once_cell::sync::Lazy;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Off,
    Bwrap,
    Unshare,
}

impl Kind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Kind::Off => "off",
            Kind::Bwrap => "bwrap",
            Kind::Unshare => "unshare",
        }
    }
}

fn works(c: &mut Command) -> bool {
    c.stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).status().map(|s| s.success()).unwrap_or(false)
}

static BWRAP_OK: Lazy<bool> = Lazy::new(|| {
    crate::util::which("bwrap").is_some()
        && works(Command::new("bwrap").args(["--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--unshare-net", "--die-with-parent", "true"]))
});
static UNSHARE_OK: Lazy<bool> = Lazy::new(|| {
    crate::util::which("unshare").is_some() && works(Command::new("unshare").args(["--user", "--map-root-user", "--mount", "--net", "true"]))
});

/// The backend a preference resolves to on this machine.
pub fn resolve(pref: &str) -> Kind {
    match pref.trim().to_lowercase().as_str() {
        "off" | "none" => Kind::Off,
        "bwrap" => if *BWRAP_OK { Kind::Bwrap } else { Kind::Off },
        "unshare" => if *UNSHARE_OK { Kind::Unshare } else { Kind::Off },
        _ => {
            if *BWRAP_OK {
                Kind::Bwrap
            } else if *UNSHARE_OK {
                Kind::Unshare
            } else {
                Kind::Off
            }
        }
    }
}

/// What is available, for the Doctor and the Overview.
pub fn availability() -> (bool, bool) {
    (*BWRAP_OK, *UNSHARE_OK)
}

#[derive(Debug, Clone)]
pub struct Sandbox {
    pub kind: Kind,
    pub workspace: PathBuf,
    /// Absolute paths inside the workspace that stay read-only.
    pub readonly: Vec<PathBuf>,
    pub network: bool,
}

fn sh_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', r"'\''"))
}

/// Does this environment variable look like a credential? Agent commands never see these.
pub fn is_secret_var(name: &str) -> bool {
    let n = name.to_ascii_uppercase();
    const MARKS: [&str; 9] = ["KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "AUTH", "COOKIE", "SESSION"];
    MARKS.iter().any(|m| n.contains(m)) || n.starts_with("HARNESS2_") || n.starts_with("AWS_") || n == "DATABASE_URL"
}

/// Remove credential-looking variables from a command the model controls (upstream #37/#40):
/// a prompt-injected agent could otherwise print the server's API keys with `env`. Names listed
/// in HARNESS2_PASS_ENV (comma-separated) are kept, for goals that really need one.
pub fn scrub_env(c: &mut Command) {
    let keep: Vec<String> = std::env::var("HARNESS2_PASS_ENV").unwrap_or_default().split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
    for (k, _) in std::env::vars_os() {
        let k = k.to_string_lossy().to_string();
        if is_secret_var(&k) && !keep.contains(&k) {
            c.env_remove(&k);
        }
    }
}

impl Sandbox {
    pub fn new(pref: &str, workspace: &Path, readonly_rel: &[String], network: bool) -> Sandbox {
        let ws = workspace.canonicalize().unwrap_or_else(|_| workspace.to_path_buf());
        let readonly = readonly_rel.iter().map(|r| ws.join(r)).filter(|p| p.exists()).collect();
        Sandbox { kind: resolve(pref), workspace: ws, readonly, network }
    }

    /// A `bash -c cmd` command, wrapped by the backend, running in the workspace, without secrets.
    pub fn command(&self, cmd: &str) -> Command {
        let mut c = self.raw_command(cmd);
        scrub_env(&mut c);
        c
    }

    fn raw_command(&self, cmd: &str) -> Command {
        let ws = self.workspace.display().to_string();
        match self.kind {
            Kind::Off => {
                let mut c = Command::new("bash");
                c.arg("-c").arg(cmd).current_dir(&self.workspace);
                c
            }
            Kind::Bwrap => {
                let mut c = Command::new("bwrap");
                c.args(["--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp", "--die-with-parent", "--unshare-pid", "--unshare-ipc"]);
                if !self.network {
                    c.arg("--unshare-net");
                }
                // Hide the home directory (keys, configs), but keep user-installed Python packages.
                if let Some(home) = std::env::var_os("HOME").map(PathBuf::from).filter(|h| h.is_dir() && h != Path::new("/")) {
                    if home != self.workspace {
                        c.arg("--tmpfs").arg(&home);
                        let local = home.join(".local");
                        if local.is_dir() {
                            c.arg("--ro-bind").arg(&local).arg(&local);
                        }
                    }
                }
                c.arg("--bind").arg(&ws).arg(&ws);
                for p in &self.readonly {
                    c.arg("--ro-bind").arg(p).arg(p);
                }
                c.arg("--chdir").arg(&ws).arg("bash").arg("-c").arg(cmd);
                c
            }
            Kind::Unshare => {
                // Inside a user + mount namespace we are root over our own mounts: make inputs
                // read-only bind mounts, then run the command.
                let mut pre = String::new();
                for p in &self.readonly {
                    let q = sh_quote(&p.display().to_string());
                    pre.push_str(&format!("mount --bind {q} {q} && mount -o remount,bind,ro {q} {q} && "));
                }
                let script = format!("{pre}cd {} && exec bash -c {}", sh_quote(&ws), sh_quote(cmd));
                let mut c = Command::new("unshare");
                c.args(["--user", "--map-root-user", "--mount"]);
                if !self.network {
                    c.arg("--net");
                }
                c.arg("bash").arg("-c").arg(script).current_dir(&self.workspace);
                c
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(sb: &Sandbox, cmd: &str) -> (bool, String) {
        let mut c = sb.command(cmd);
        let out = c.output().unwrap();
        (out.status.success(), format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr)))
    }

    #[test]
    fn secrets_are_not_passed_to_commands() {
        assert!(is_secret_var("OPENAI_API_KEY") && is_secret_var("HARNESS2_UI_PASSWORD") && is_secret_var("github_token"));
        assert!(!is_secret_var("PATH") && !is_secret_var("HOME") && !is_secret_var("LANG"));
        std::env::set_var("H2TEST_SECRET_TOKEN", "s3cr3t");
        let sb = Sandbox { kind: Kind::Off, workspace: std::env::temp_dir(), readonly: vec![], network: false };
        let (_, out) = run(&sb, "echo [$H2TEST_SECRET_TOKEN] [$PATH]");
        assert!(out.starts_with("[] [/"), "{out}");
    }

    #[test]
    fn sandbox_confines_writes_inputs_and_network() {
        for pref in ["bwrap", "unshare"] {
            if resolve(pref) == Kind::Off {
                eprintln!("{pref} not available here; skipped");
                continue;
            }
            let d = tempfile::tempdir().unwrap();
            std::fs::write(d.path().join("data.csv"), "a\n").unwrap();
            let sb = Sandbox::new(pref, d.path(), &["data.csv".into()], false);
            assert_eq!(run(&sb, "echo hi > out.txt && cat out.txt"), (true, "hi\n".into()), "{pref}: workspace is writable");
            assert!(!run(&sb, "echo x >> data.csv").0, "{pref}: inputs are read-only");
            assert_eq!(std::fs::read_to_string(d.path().join("data.csv")).unwrap(), "a\n");
            assert!(!run(&sb, "python3 -c \"import socket; socket.create_connection(('1.1.1.1', 53), 2)\"").0, "{pref}: no network");
            if pref == "bwrap" {
                assert!(!run(&sb, "touch /usr/sandbox-escape").0, "bwrap: the rest of the filesystem is read-only");
            }
        }
    }
}
