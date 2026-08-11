use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use crossterm::execute;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Gauge, List, ListItem, ListState, Paragraph};
use ratatui::Terminal;
use regex::Regex;

#[derive(Clone, Debug, PartialEq)]
enum Status {
    Completed,
    Partial,
    Pending,
    Queued,
    Downloading,
    Failed,
}

#[derive(Clone, Debug)]
struct ModelItem {
    key: String,
    name: String,
    mtype: String,
    size_str: String,
    expected_bytes: u64,
    repo_id: String,
    local_path: PathBuf,
    local_bytes: u64,
    checked: bool,
    status: Status,
    /// (missing_files, total_files) from remote verification
    files: Option<(usize, usize)>,
}

fn parse_size(s: &str) -> u64 {
    // "~60GB", "1.2GB", "3GB", "167MB"
    let re = Regex::new(r"([\d.]+)\s*(GB|MB)").unwrap();
    if let Some(c) = re.captures(s) {
        let n: f64 = c[1].parse().unwrap_or(0.0);
        let mult = if &c[2] == "GB" { 1_000_000_000.0 } else { 1_000_000.0 };
        (n * mult) as u64
    } else {
        0
    }
}

fn dir_size(path: &Path) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(path) {
        for e in entries.flatten() {
            let p = e.path();
            if let Ok(md) = e.metadata() {
                if md.is_dir() {
                    total += dir_size(&p);
                } else {
                    total += md.len();
                }
            }
        }
    }
    total
}

fn human(bytes: u64) -> String {
    let b = bytes as f64;
    if b >= 1e9 {
        format!("{:.1}GB", b / 1e9)
    } else if b >= 1e6 {
        format!("{:.1}MB", b / 1e6)
    } else if b >= 1e3 {
        format!("{:.1}KB", b / 1e3)
    } else {
        format!("{}B", bytes)
    }
}

fn parse_models_py(root: &Path) -> Vec<ModelItem> {
    let src = std::fs::read_to_string(root.join("models.py")).expect("cannot read models.py");

    // Parse MODEL_PATHS
    let mut paths: HashMap<String, String> = HashMap::new();
    let re_paths = Regex::new(r#""(\w+)":\s*"(models/[\w/]+)""#).unwrap();
    for c in re_paths.captures_iter(&src) {
        paths.insert(c[1].to_string(), c[2].to_string());
    }

    // Parse MODEL_METADATA blocks
    let re_block = Regex::new(
        r#"(?s)"(\w+)":\s*\{\s*"name":\s*"([^"]+)",\s*"type":\s*"(\w+)",\s*"size":\s*"([^"]+)",\s*"description":\s*"[^"]*",\s*"repo_id":\s*"([^"]+)""#,
    )
    .unwrap();

    let mut items = Vec::new();
    for c in re_block.captures_iter(&src) {
        let key = c[1].to_string();
        let mtype = c[3].to_string();
        let base = paths
            .get(&mtype)
            .cloned()
            .unwrap_or_else(|| format!("models/{}", mtype));
        let local_path = root.join(&base).join(&key);
        let size_str = c[4].to_string();
        let expected = parse_size(&size_str);
        let local = if local_path.exists() { dir_size(&local_path) } else { 0 };
        let status = if local == 0 {
            Status::Pending
        } else if expected > 0 && (local as f64) < expected as f64 * 0.85 {
            Status::Partial
        } else {
            Status::Completed
        };
        items.push(ModelItem {
            key,
            name: c[2].to_string(),
            mtype,
            size_str,
            expected_bytes: expected,
            repo_id: c[5].to_string(),
            local_path,
            local_bytes: local,
            checked: false,
            status,
            files: None,
        });
    }
    items
}

fn hf_binary(root: &Path) -> PathBuf {
    if let Ok(p) = std::env::var("HF_BIN") {
        return PathBuf::from(p);
    }
    let venv = root.join(".venv/bin/hf");
    if venv.exists() {
        venv
    } else {
        PathBuf::from("hf")
    }
}

enum DlMsg {
    Line(usize, String),
    Done(usize, bool),
    /// Result of remote verification: (idx, missing_files, total_files, exact_total_bytes)
    Verified(usize, usize, usize, u64),
    VerifyFailed(usize, String),
}

fn hf_token() -> Option<String> {
    if let Ok(t) = std::env::var("HF_TOKEN") {
        return Some(t);
    }
    let home = std::env::var("HOME").ok()?;
    for p in [".cache/huggingface/token", ".huggingface/token"] {
        if let Ok(t) = std::fs::read_to_string(Path::new(&home).join(p)) {
            let t = t.trim().to_string();
            if !t.is_empty() {
                return Some(t);
            }
        }
    }
    None
}

/// Files install.py deliberately skips; ignore them when verifying too.
fn ignored_file(name: &str) -> bool {
    let n = name.to_lowercase();
    [".jpg", ".jpeg", ".png", ".gif", ".md"]
        .iter()
        .any(|ext| n.ends_with(ext))
}

/// Fetch the repo file list (name, size) from the HF API.
fn remote_files(repo_id: &str) -> Result<Vec<(String, u64)>, String> {
    let url = format!("https://huggingface.co/api/models/{repo_id}?blobs=true");
    let mut req = ureq::get(&url).timeout(Duration::from_secs(20));
    if let Some(t) = hf_token() {
        req = req.set("Authorization", &format!("Bearer {t}"));
    }
    let body = req
        .call()
        .map_err(|e| format!("{e}"))?
        .into_string()
        .map_err(|e| format!("{e}"))?;
    let v: serde_json::Value = serde_json::from_str(&body).map_err(|e| format!("{e}"))?;
    let sib = v["siblings"].as_array().ok_or("no siblings in response")?;
    Ok(sib
        .iter()
        .filter_map(|s| {
            Some((
                s["rfilename"].as_str()?.to_string(),
                s["size"].as_u64().unwrap_or(0),
            ))
        })
        .filter(|(name, _)| !ignored_file(name))
        .collect())
}

/// Verify all models against the HF API in a background thread.
fn spawn_verify(targets: Vec<(usize, String, PathBuf)>, tx: Sender<DlMsg>) {
    thread::spawn(move || {
        for (idx, repo_id, local) in targets {
            match remote_files(&repo_id) {
                Ok(files) => {
                    let total = files.len();
                    let exact: u64 = files.iter().map(|(_, s)| s).sum();
                    let missing = files
                        .iter()
                        .filter(|(name, size)| {
                            let p = local.join(name);
                            match p.metadata() {
                                Ok(md) => *size > 0 && md.len() != *size,
                                Err(_) => true,
                            }
                        })
                        .count();
                    let _ = tx.send(DlMsg::Verified(idx, missing, total, exact));
                }
                Err(e) => {
                    let _ = tx.send(DlMsg::VerifyFailed(idx, e));
                }
            }
        }
    });
}

/// Where the hf CLI stages data while downloading (xet/hub caches).
fn hf_cache_dirs(repo_id: &str) -> Vec<PathBuf> {
    let base = std::env::var("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".cache/huggingface")
        });
    vec![
        base.join("hub")
            .join(format!("models--{}", repo_id.replace('/', "--"))),
        base.join("xet"),
    ]
}

/// Try to extract a transfer speed like "45.3MB/s" from an hf output line.
fn parse_speed(line: &str) -> Option<f64> {
    let re = Regex::new(r"([\d.]+)\s*([kKMG]?)i?B/s").unwrap();
    let c = re.captures(line)?;
    let n: f64 = c[1].parse().ok()?;
    let mult = match &c[2] {
        "k" | "K" => 1e3,
        "M" => 1e6,
        "G" => 1e9,
        _ => 1.0,
    };
    Some(n * mult)
}

fn spawn_download(
    idx: usize,
    repo_id: String,
    dest: PathBuf,
    hf: PathBuf,
    tx: Sender<DlMsg>,
    pid_slot: Arc<Mutex<Option<u32>>>,
    log_path: PathBuf,
) {
    thread::spawn(move || {
        use std::io::Write;
        if let Some(dir) = log_path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let log = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
            .ok()
            .map(|f| Arc::new(Mutex::new(f)));
        let log_line = |s: &str| {
            if let Some(l) = &log {
                let ts = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                let _ = writeln!(l.lock().unwrap(), "[{ts}] {s}");
            }
        };
        log_line(&format!("=== starting: hf download {repo_id} --local-dir {}", dest.display()));
        let child = Command::new(&hf)
            .args(["download", &repo_id, "--local-dir"])
            .arg(&dest)
            .stdout(Stdio::piped())
            .env("PYTHONUNBUFFERED", "1")
            .env("HF_HUB_DISABLE_PROGRESS_BARS", "0")
            .env("FORCE_COLOR", "0")
            .stderr(Stdio::piped())
            .spawn();
        let mut child = match child {
            Ok(c) => c,
            Err(e) => {
                log_line(&format!("failed to start hf: {e}"));
                let _ = tx.send(DlMsg::Line(idx, format!("failed to start hf: {e}")));
                let _ = tx.send(DlMsg::Done(idx, false));
                return;
            }
        };
        *pid_slot.lock().unwrap() = Some(child.id());
        // Read stdout in the background so the pipe can't fill up and block hf
        let stdout = child.stdout.take().unwrap();
        let log_out = log.clone();
        let h_out = thread::spawn(move || {
            use std::io::{BufRead, BufReader};
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                if let Some(l) = &log_out {
                    let _ = writeln!(l.lock().unwrap(), "[stdout] {line}");
                }
            }
        });
        // Read stderr (progress) byte-by-byte, splitting on \r or \n
        let stderr = child.stderr.take().unwrap();
        let tx2 = tx.clone();
        let log_err = log.clone();
        let h = thread::spawn(move || {
            use std::io::Read;
            let mut buf = [0u8; 4096];
            let mut line = Vec::new();
            let mut r = stderr;
            loop {
                match r.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => {
                        for &b in &buf[..n] {
                            if b == b'\r' || b == b'\n' {
                                if !line.is_empty() {
                                    let s = String::from_utf8_lossy(&line).trim().to_string();
                                    if !s.is_empty() {
                                        if let Some(l) = &log_err {
                                            let _ = writeln!(l.lock().unwrap(), "{s}");
                                        }
                                        let _ = tx2.send(DlMsg::Line(idx, s));
                                    }
                                    line.clear();
                                }
                            } else {
                                line.push(b);
                            }
                        }
                    }
                }
            }
        });
        let status = child.wait();
        let ok = status.as_ref().map(|s| s.success()).unwrap_or(false);
        log_line(&format!("=== finished: {:?}", status));
        *pid_slot.lock().unwrap() = None;
        let _ = h.join();
        let _ = h_out.join();
        let _ = tx.send(DlMsg::Done(idx, ok));
    });
}

struct App {
    items: Vec<ModelItem>,
    list_state: ListState,
    downloading: Option<usize>,
    queue: Vec<usize>,
    log_line: String,
    rx: Receiver<DlMsg>,
    tx: Sender<DlMsg>,
    hf: PathBuf,
    last_refresh: std::time::Instant,
    /// When we last got a real line from hf's output
    last_hf_line: Option<std::time::Instant>,
    /// When hf last reported its own transfer speed
    hf_speed_at: Option<std::time::Instant>,
    /// Combined bytes (local + caches) at the start of the active download
    dl_start_bytes: Option<u64>,
    /// (timestamp, bytes) samples of the active download, for speed calc
    samples: Vec<(std::time::Instant, u64)>,
    speed_bps: f64,
    active_pid: Arc<Mutex<Option<u32>>>,
    cancelling: bool,
    log_dir: PathBuf,
}

impl App {
    fn new(root: &Path) -> Self {
        let (tx, rx) = mpsc::channel();
        let mut ls = ListState::default();
        ls.select(Some(0));
        let items = parse_models_py(root);
        let targets: Vec<(usize, String, PathBuf)> = items
            .iter()
            .enumerate()
            .map(|(i, it)| (i, it.repo_id.clone(), it.local_path.clone()))
            .collect();
        spawn_verify(targets, tx.clone());
        App {
            items,
            list_state: ls,
            downloading: None,
            queue: Vec::new(),
            log_line: String::new(),
            rx,
            tx,
            hf: hf_binary(root),
            last_refresh: std::time::Instant::now(),
            last_hf_line: None,
            hf_speed_at: None,
            dl_start_bytes: None,
            samples: Vec::new(),
            speed_bps: 0.0,
            active_pid: Arc::new(Mutex::new(None)),
            cancelling: false,
            log_dir: root.join("logs/models-ui"),
        }
    }

    fn log_path_for(&self, idx: usize) -> PathBuf {
        self.log_dir.join(format!("{}.log", self.items[idx].key))
    }

    /// Detect competing `hf download` processes for the same repo, which
    /// cause lock-timeout failures.
    fn competing_download(&self, repo_id: &str) -> Option<String> {
        let out = Command::new("pgrep").args(["-fl", "hf download"]).output().ok()?;
        let s = String::from_utf8_lossy(&out.stdout);
        s.lines()
            .find(|l| l.contains(repo_id) && !l.contains(&std::process::id().to_string()))
            .map(|l| l.split_whitespace().next().unwrap_or("?").to_string())
    }

    fn start_next(&mut self) {
        if self.downloading.is_some() {
            return;
        }
        if let Some(idx) = self.queue.first().copied() {
            self.queue.remove(0);
            if let Some(pid) = self.competing_download(&self.items[idx].repo_id) {
                self.items[idx].status = Status::Failed;
                self.items[idx].checked = false;
                self.log_line = format!(
                    "Another 'hf download' for {} is already running (pid {}) - kill it first",
                    self.items[idx].repo_id, pid
                );
                return;
            }
            self.items[idx].status = Status::Downloading;
            self.downloading = Some(idx);
            let log_path = self.log_path_for(idx);
            spawn_download(
                idx,
                self.items[idx].repo_id.clone(),
                self.items[idx].local_path.clone(),
                self.hf.clone(),
                self.tx.clone(),
                self.active_pid.clone(),
                log_path,
            );
        }
    }

    /// Stop the active download (keeps queue intact; press 'd' to resume it).
    fn cancel_active(&mut self) {
        if let Some(pid) = *self.active_pid.lock().unwrap() {
            self.cancelling = true;
            let _ = Command::new("kill").arg(pid.to_string()).status();
            self.log_line = "Cancelling download...".into();
        }
    }

    fn tick(&mut self) {
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                DlMsg::Line(idx, s) => {
                    // Prefer the speed hf itself reports, when present
                    if let Some(sp) = parse_speed(&s) {
                        self.speed_bps = sp;
                        self.hf_speed_at = Some(std::time::Instant::now());
                    }
                    let _ = idx;
                    self.log_line = s;
                    self.last_hf_line = Some(std::time::Instant::now());
                }
                DlMsg::Done(idx, ok) => {
                    let cancelled = self.cancelling;
                    self.cancelling = false;
                    let log_hint = self.log_path_for(idx);
                    let it = &mut self.items[idx];
                    it.local_bytes = dir_size(&it.local_path);
                    it.status = if ok {
                        Status::Completed
                    } else if cancelled {
                        self.log_line = "Download stopped - check + 'd' to resume".into();
                        if it.local_bytes == 0 { Status::Pending } else { Status::Partial }
                    } else {
                        self.log_line =
                            format!("Download failed - see {}", log_hint.display());
                        Status::Failed
                    };
                    it.checked = false;
                    self.downloading = None;
                    self.samples.clear();
                    self.speed_bps = 0.0;
                    self.hf_speed_at = None;
                    self.dl_start_bytes = None;
                    // Don't auto-continue the queue after a manual stop
                    if !cancelled {
                        self.start_next();
                    }
                    // Re-verify against the HF API after any download ends
                    spawn_verify(
                        vec![(
                            idx,
                            self.items[idx].repo_id.clone(),
                            self.items[idx].local_path.clone(),
                        )],
                        self.tx.clone(),
                    );
                }
                DlMsg::Verified(idx, missing, total, exact_bytes) => {
                    let it = &mut self.items[idx];
                    if exact_bytes > 0 {
                        it.expected_bytes = exact_bytes;
                    }
                    it.files = Some((missing, total));
                    if !matches!(it.status, Status::Downloading | Status::Queued) {
                        it.status = if missing == 0 {
                            Status::Completed
                        } else if it.local_bytes == 0 {
                            Status::Pending
                        } else {
                            Status::Partial
                        };
                    }
                }
                DlMsg::VerifyFailed(idx, e) => {
                    self.log_line =
                        format!("verify {}: {} (using size heuristic)", self.items[idx].key, e);
                }
            }
        }
        // Poll size of the active download every ~2s and update speed
        if self.last_refresh.elapsed() > Duration::from_secs(2) {
            if let Some(idx) = self.downloading {
                let it = &mut self.items[idx];
                it.local_bytes = dir_size(&it.local_path);
                // hf (xet backend) stages data in the HF caches, not the
                // local dir, so measure everything it could be writing to.
                let mut total = it.local_bytes;
                for d in hf_cache_dirs(&it.repo_id) {
                    total += dir_size(&d);
                }
                let now = std::time::Instant::now();
                let base = *self.dl_start_bytes.get_or_insert(total);
                self.samples.push((now, total));
                // Keep a ~30s sliding window
                self.samples
                    .retain(|(t, _)| now.duration_since(*t) < Duration::from_secs(30));
                // Only fall back to disk-based speed if hf hasn't reported
                // its own speed recently.
                let hf_speed_fresh = self
                    .hf_speed_at
                    .map(|t| t.elapsed() < Duration::from_secs(10))
                    .unwrap_or(false);
                if !hf_speed_fresh && self.samples.len() >= 2 {
                    let (t0, b0) = self.samples[0];
                    let (t1, b1) = *self.samples.last().unwrap();
                    let dt = t1.duration_since(t0).as_secs_f64();
                    if dt > 0.5 {
                        self.speed_bps = (b1.saturating_sub(b0)) as f64 / dt;
                    }
                }
                // If hf has been quiet for >5s, synthesize a status line so the
                // user always sees movement (hf hides progress bars on non-TTY).
                let hf_quiet = self
                    .last_hf_line
                    .map(|t| t.elapsed() > Duration::from_secs(5))
                    .unwrap_or(true);
                if hf_quiet {
                    self.log_line = format!(
                        "{}: {} transferred this session @ {}/s",
                        self.items[idx].name,
                        human(total.saturating_sub(base)),
                        human(self.speed_bps as u64)
                    );
                }
            }
            self.last_refresh = std::time::Instant::now();
        }
    }
}

fn main() -> io::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let list_mode = args.iter().any(|a| a == "--list");
    let root = args
        .iter()
        .find(|a| !a.starts_with("--"))
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().unwrap());

    if list_mode {
        for it in parse_models_py(&root) {
            let (status, files) = match remote_files(&it.repo_id) {
                Ok(fs) => {
                    let missing = fs
                        .iter()
                        .filter(|(name, size)| {
                            let p = it.local_path.join(name);
                            match p.metadata() {
                                Ok(md) => *size > 0 && md.len() != *size,
                                Err(_) => true,
                            }
                        })
                        .count();
                    let st = if missing == 0 {
                        Status::Completed
                    } else if it.local_bytes == 0 {
                        Status::Pending
                    } else {
                        Status::Partial
                    };
                    (st, format!("{}/{} files", fs.len() - missing, fs.len()))
                }
                Err(_) => (it.status.clone(), "unverified".into()),
            };
            println!(
                "{:<12} {:<24} {:<16} {:>8} / {:<8} {:<14} {}",
                format!("{:?}", status),
                it.name,
                it.mtype,
                human(it.local_bytes),
                it.size_str,
                files,
                it.repo_id
            );
        }
        return Ok(());
    }

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new(&root);
    let res = run(&mut terminal, &mut app);

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    res
}

fn run(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>, app: &mut App) -> io::Result<()> {
    loop {
        app.tick();
        terminal.draw(|f| ui(f, app))?;

        if event::poll(Duration::from_millis(200))? {
            if let Event::Key(k) = event::read()? {
                if k.kind != KeyEventKind::Press {
                    continue;
                }
                let sel = app.list_state.selected().unwrap_or(0);
                match k.code {
                    KeyCode::Char('q') | KeyCode::Esc => {
                        if app.downloading.is_none() {
                            return Ok(());
                        }
                        app.log_line =
                            "Download in progress - press Shift+Q to force quit".into();
                    }
                    KeyCode::Char('Q') => return Ok(()),
                    KeyCode::Down | KeyCode::Char('j') => {
                        let n = app.items.len();
                        app.list_state.select(Some((sel + 1) % n));
                    }
                    KeyCode::Up | KeyCode::Char('k') => {
                        let n = app.items.len();
                        app.list_state.select(Some((sel + n - 1) % n));
                    }
                    KeyCode::Char(' ') => {
                        let it = &mut app.items[sel];
                        if matches!(it.status, Status::Pending | Status::Partial | Status::Failed) {
                            it.checked = !it.checked;
                        }
                    }
                    KeyCode::Char('d') | KeyCode::Enter => {
                        for (i, it) in app.items.iter_mut().enumerate() {
                            if it.checked
                                && !app.queue.contains(&i)
                                && app.downloading != Some(i)
                                && matches!(it.status, Status::Pending | Status::Partial | Status::Failed)
                            {
                                it.status = Status::Queued;
                                app.queue.push(i);
                            }
                        }
                        app.start_next();
                    }
                    KeyCode::Char('x') => {
                        app.cancel_active();
                    }
                    KeyCode::Char('r') => {
                        for it in app.items.iter_mut() {
                            if matches!(it.status, Status::Downloading | Status::Queued) {
                                continue;
                            }
                            it.local_bytes = dir_size(&it.local_path);
                            it.status = if it.local_bytes == 0 {
                                Status::Pending
                            } else if it.expected_bytes > 0
                                && (it.local_bytes as f64) < it.expected_bytes as f64 * 0.85
                            {
                                Status::Partial
                            } else {
                                Status::Completed
                            };
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}

fn ui(f: &mut ratatui::Frame, app: &mut App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(5),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(1),
        ])
        .split(f.area());

    let items: Vec<ListItem> = app
        .items
        .iter()
        .map(|it| {
            let (mark, color) = match it.status {
                Status::Completed => ("✔ completed ".to_string(), Color::Green),
                Status::Downloading => ("↓ downloading".to_string(), Color::Cyan),
                Status::Queued => ("… queued    ".to_string(), Color::Yellow),
                Status::Failed => ("✗ failed    ".to_string(), Color::Red),
                Status::Partial => (
                    format!("[{}] partial  ", if it.checked { "x" } else { " " }),
                    Color::Magenta,
                ),
                Status::Pending => (
                    format!("[{}] pending  ", if it.checked { "x" } else { " " }),
                    Color::White,
                ),
            };
            let line = Line::from(vec![
                Span::styled(format!("{mark}  "), Style::default().fg(color)),
                Span::styled(
                    format!("{:<24}", it.name),
                    Style::default().add_modifier(Modifier::BOLD),
                ),
                Span::raw(format!(
                    " {:<16} {:>8} / {:<8}  {:<12} {}",
                    it.mtype,
                    human(it.local_bytes),
                    if it.files.is_some() {
                        human(it.expected_bytes)
                    } else {
                        it.size_str.clone()
                    },
                    match it.files {
                        Some((0, n)) => format!("{n}/{n} files"),
                        Some((m, n)) => format!("{}/{} files", n - m, n),
                        None => "verifying…".to_string(),
                    },
                    it.repo_id
                )),
            ]);
            ListItem::new(line)
        })
        .collect();

    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(" Story Engine — Models "))
        .highlight_style(Style::default().bg(Color::DarkGray))
        .highlight_symbol("> ");
    f.render_stateful_widget(list, chunks[0], &mut app.list_state);

    // Progress gauge for active download
    let gauge_area: Rect = chunks[1];
    if let Some(idx) = app.downloading {
        let it = &app.items[idx];
        let ratio = if it.expected_bytes > 0 {
            (it.local_bytes as f64 / it.expected_bytes as f64).min(1.0)
        } else {
            0.0
        };
        let speed = if app.speed_bps > 0.0 {
            let eta = if it.expected_bytes > it.local_bytes {
                let secs = (it.expected_bytes - it.local_bytes) as f64 / app.speed_bps;
                if secs >= 3600.0 {
                    format!("  ETA {:.1}h", secs / 3600.0)
                } else if secs >= 60.0 {
                    format!("  ETA {:.0}m", secs / 60.0)
                } else {
                    format!("  ETA {:.0}s", secs)
                }
            } else {
                String::new()
            };
            format!(" @ {}/s{}", human(app.speed_bps as u64), eta)
        } else {
            String::new()
        };
        let g = Gauge::default()
            .block(Block::default().borders(Borders::ALL).title(format!(
                " Downloading {} ({} of {}){} ",
                it.name,
                human(it.local_bytes),
                it.size_str,
                speed
            )))
            .gauge_style(Style::default().fg(Color::Cyan))
            .ratio(ratio);
        f.render_widget(g, gauge_area);
    } else {
        let p = Paragraph::new("No active download")
            .block(Block::default().borders(Borders::ALL).title(" Progress "));
        f.render_widget(p, gauge_area);
    }

    let log = Paragraph::new(app.log_line.as_str())
        .block(Block::default().borders(Borders::ALL).title(" hf output "));
    f.render_widget(log, chunks[2]);

    let help = Paragraph::new("↑/↓ move   SPACE check   d/ENTER download   x stop download   r refresh   q quit")
        .style(Style::default().fg(Color::DarkGray));
    f.render_widget(help, chunks[3]);
}
