//! `lucy sweep`: run seeds x parameter grids as separate `lucy run` processes, `jobs` at a time,
//! then summarize each run's last report windows in one table (and summary.json).

use std::fs;
use std::path::Path;
use std::process::Command;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, Result, bail, ensure};
use serde_json::{Map, Value, json};

/// One run of the sweep.
struct Job {
    name: String,
    seed: u64,
    params: Map<String, Value>,
}

/// `name=v1,v2,...` -> (name, [v1, v2, ...]) with numeric values.
fn parse_axis(s: &str) -> Result<(String, Vec<f64>)> {
    let (name, vals) = s
        .split_once('=')
        .with_context(|| format!("--param {s}: expected name=v1,v2"))?;
    let vals = vals
        .split(',')
        .map(|v| {
            v.trim()
                .parse::<f64>()
                .with_context(|| format!("--param {s}: bad number {v}"))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((name.trim().to_string(), vals))
}

fn jobs(seeds: &[u64], axes: &[(String, Vec<f64>)], base: &Map<String, Value>) -> Vec<Job> {
    let mut grid: Vec<Vec<(String, f64)>> = vec![vec![]];
    for (name, vals) in axes {
        grid = grid
            .into_iter()
            .flat_map(|point| {
                vals.iter().map(move |&v| {
                    let mut p = point.clone();
                    p.push((name.clone(), v));
                    p
                })
            })
            .collect();
    }
    let mut out = Vec::new();
    for &seed in seeds {
        for point in &grid {
            let mut params = base.clone();
            let mut name = format!("s{seed}");
            for (k, v) in point {
                // Integral values as JSON integers, so integer params (e.g. periods) parse.
                let val = if v.fract() == 0.0 && v.abs() < 1e15 {
                    json!(*v as i64)
                } else {
                    json!(v)
                };
                params.insert(k.clone(), val);
                name.push_str(&format!("_{k}={v}"));
            }
            out.push(Job { name, seed, params });
        }
    }
    out
}

/// Mean of `f` over the last `n` windows where it is defined.
fn tail_mean(rows: &[Value], n: usize, f: impl Fn(&Value) -> Option<f64>) -> Option<f64> {
    let xs: Vec<f64> = rows.iter().rev().filter_map(&f).take(n).collect();
    (!xs.is_empty()).then(|| xs.iter().sum::<f64>() / xs.len() as f64)
}

fn summarize(path: &Path) -> Result<Value> {
    let text = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let rows: Vec<Value> = text
        .lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .filter(|v| v["kind"] == "metrics" && !v["accuracy"].is_null())
        .collect();
    let f = |key: &'static str| tail_mean(&rows, 10, |r| r[key].as_f64());
    let base = |i: usize| tail_mean(&rows, 10, |r| r["baseline_accuracy"][i].as_f64());
    Ok(json!({
        "windows": rows.len(),
        "accuracy": f("accuracy"),
        "surprise_bits": f("surprise_bits"),
        "unigram_acc": base(0),
        "bigram_acc": base(1),
        "trigram_acc": base(2),
        "fires_per_tick": f("fires_per_tick"),
        "reverb_fraction": f("reverb_fraction"),
        "theta": tail_mean(&rows, 10, |r| r["reservoir"]["theta_mean"].as_f64()),
        "events_per_sec": tail_mean(&rows, 10, |r| r["wall"]["events_per_sec"].as_f64()),
    }))
}

pub struct Sweep<'a> {
    /// Arguments forwarded to every `lucy run` (corpus, ticks, graph size, ...).
    pub run_args: Vec<String>,
    pub seeds: &'a [u64],
    pub axes: &'a [String],
    pub base_params: Option<&'a Path>,
    pub jobs: usize,
    pub out: &'a Path,
}

impl Sweep<'_> {
    pub fn run(&self) -> Result<()> {
        ensure!(self.jobs >= 1, "--jobs must be >= 1");
        ensure!(!self.seeds.is_empty(), "--seeds is empty");
        let axes = self
            .axes
            .iter()
            .map(|a| parse_axis(a))
            .collect::<Result<Vec<_>>>()?;
        let base = match self.base_params {
            Some(p) => serde_json::from_str::<Map<String, Value>>(&fs::read_to_string(p)?)
                .with_context(|| format!("parsing {}", p.display()))?,
            None => Map::new(),
        };
        let jobs = jobs(self.seeds, &axes, &base);
        fs::create_dir_all(self.out).with_context(|| format!("creating {}", self.out.display()))?;
        let exe = std::env::current_exe().context("locating the lucy binary")?;
        println!(
            "sweep: {} runs, {} at a time -> {}",
            jobs.len(),
            self.jobs,
            self.out.display()
        );

        let next = AtomicUsize::new(0);
        let failures = Mutex::new(Vec::new());
        std::thread::scope(|s| {
            for _ in 0..self.jobs.min(jobs.len()) {
                s.spawn(|| {
                    while let Some(job) = jobs.get(next.fetch_add(1, Ordering::Relaxed)) {
                        if let Err(e) = self.launch(&exe, job) {
                            failures
                                .lock()
                                .expect("lock")
                                .push(format!("{}: {e:#}", job.name));
                        }
                    }
                });
            }
        });
        let failures = failures.into_inner().expect("lock");

        let mut table = Vec::new();
        println!(
            "\n{:<36} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>9}",
            "run", "acc", "uni", "bigram", "trigram", "surpr", "f/tick", "reverb", "theta", "ev/s"
        );
        for job in &jobs {
            let Ok(s) = summarize(&self.out.join(format!("{}.jsonl", job.name))) else {
                continue;
            };
            let g = |k: &str| s[k].as_f64().unwrap_or(f64::NAN);
            println!(
                "{:<36} {:>7.4} {:>7.4} {:>7.4} {:>7.4} {:>7.3} {:>7.1} {:>7.3} {:>7.4} {:>9.0}",
                job.name,
                g("accuracy"),
                g("unigram_acc"),
                g("bigram_acc"),
                g("trigram_acc"),
                g("surprise_bits"),
                g("fires_per_tick"),
                g("reverb_fraction"),
                g("theta"),
                g("events_per_sec")
            );
            table.push(
                json!({"run": job.name, "seed": job.seed, "params": job.params, "summary": s}),
            );
        }
        fs::write(
            self.out.join("summary.json"),
            serde_json::to_string_pretty(&table)?,
        )?;
        if !failures.is_empty() {
            bail!("{} run(s) failed:\n{}", failures.len(), failures.join("\n"));
        }
        Ok(())
    }

    fn launch(&self, exe: &Path, job: &Job) -> Result<()> {
        let params = self.out.join(format!("{}.params.json", job.name));
        fs::write(&params, serde_json::to_string(&job.params)?)?;
        let log = fs::File::create(self.out.join(format!("{}.out", job.name)))?;
        let status = Command::new(exe)
            .arg("run")
            .args(&self.run_args)
            .arg("--seed")
            .arg(job.seed.to_string())
            .arg("--params")
            .arg(&params)
            .arg("--metrics")
            .arg(self.out.join(format!("{}.jsonl", job.name)))
            .stdout(log.try_clone()?)
            .stderr(log)
            .status()
            .context("spawning lucy run")?;
        ensure!(status.success(), "exit status {status}");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_is_seeds_times_cartesian_product() {
        let axes = vec![
            parse_axis("eta_g=0.01,0.05").unwrap(),
            parse_axis("t_heat=4").unwrap(),
        ];
        let js = jobs(&[1, 2], &axes, &Map::new());
        let names: Vec<_> = js.iter().map(|j| j.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "s1_eta_g=0.01_t_heat=4",
                "s1_eta_g=0.05_t_heat=4",
                "s2_eta_g=0.01_t_heat=4",
                "s2_eta_g=0.05_t_heat=4"
            ]
        );
        assert_eq!(js[1].params["eta_g"], json!(0.05));
        assert!(parse_axis("eta_g").is_err() && parse_axis("eta_g=x").is_err());
    }
}
