//! Soak tests: hold a steady load for hours and ask whether the server gets
//! worse with time (a memory leak, a cache that fills up, thermal throttling,
//! a connection pool that runs dry).
//!
//! A soak run is a series of equal windows, one report step each (`soak-001`,
//! `soak-002`, …). [`analyze`] compares how the first windows did with the last
//! ones and says whether the server held up. Everything is arithmetic on the
//! step results, so it is the same for a library caller, the CLI and a replay of
//! a stored report.

use crate::writers::BenchmarkResultsWriter;
use serde::{Deserialize, Serialize};

/// Id prefix of a soak window's step.
pub const WINDOW_PREFIX: &str = "soak-";
/// How much worse the last windows may be than the first ones, as a fraction.
pub const DEFAULT_TOLERANCE: f64 = 0.2;
/// Windows needed before a trend means anything.
pub const MIN_WINDOWS: usize = 3;
/// How much the share of failed requests may grow (0.01 is one point).
pub const ERROR_RATE_RISE: f64 = 0.01;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SoakVerdict {
    /// The last windows are as good as the first ones, within the tolerance.
    Stable,
    /// At least one measure got worse than the tolerance allows.
    Degrading,
    /// Fewer than [`MIN_WINDOWS`] windows had requests: too short to say.
    TooShort,
}

/// One measure, first windows against last windows.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DriftMetric {
    pub id: String,
    pub label: String,
    pub unit: String,
    /// Median of the first third of the windows.
    pub first: f64,
    /// Median of the last third of the windows.
    pub last: f64,
    /// `(last - first) / first`; absent when `first` is 0.
    pub change: Option<f64>,
    /// Change per hour of soak, from a straight line through every window.
    pub slope_per_hour: Option<f64>,
    /// Whether the change is in the bad direction by more than the limit.
    pub worse: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SoakSummary {
    pub verdict: SoakVerdict,
    /// Windows with requests that were compared.
    pub windows: usize,
    pub window_secs: f64,
    pub tolerance: f64,
    pub metrics: Vec<DriftMetric>,
    /// Why the verdict is what it is, one line each.
    pub reasons: Vec<String>,
}

/// Whether a bigger number is worse, and how a worsening is judged.
#[derive(Clone, Copy)]
enum Limit {
    /// Latencies: worse when they grew by more than the tolerance.
    Grew,
    /// Throughput: worse when it fell by more than the tolerance.
    Fell,
    /// Rates between 0 and 1: worse when they rose by more than [`ERROR_RATE_RISE`].
    Rose,
}

struct Measure {
    id: &'static str,
    label: &'static str,
    unit: &'static str,
    limit: Limit,
    read: fn(&BenchmarkResultsWriter) -> f64,
}

const MEASURES: [Measure; 5] = [
    Measure {
        id: "e2e_p95",
        label: "Total time (slowest 5%)",
        unit: "ms",
        limit: Limit::Grew,
        read: |r| r.e2e_latency_ms.p95,
    },
    Measure {
        id: "ttft_p95",
        label: "First output (slowest 5%)",
        unit: "ms",
        limit: Limit::Grew,
        read: |r| r.time_to_first_token_ms.p95,
    },
    Measure {
        id: "itl_p95",
        label: "Between outputs (slowest 5%)",
        unit: "ms",
        limit: Limit::Grew,
        read: |r| r.inter_token_latency_ms.p95,
    },
    Measure {
        id: "rps",
        label: "Requests per second",
        unit: "req/s",
        limit: Limit::Fell,
        read: |r| r.request_rate,
    },
    Measure {
        id: "error_rate",
        label: "Failed requests",
        unit: "share",
        limit: Limit::Rose,
        read: |r| r.error_rate,
    },
];

fn median(values: &[f64]) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    match v.len() {
        0 => 0.0,
        n if n % 2 == 1 => v[n / 2],
        n => (v[n / 2 - 1] + v[n / 2]) / 2.0,
    }
}

/// Least-squares slope of `values` over `hours` (the same length).
fn slope(values: &[f64], hours: &[f64]) -> Option<f64> {
    let n = values.len() as f64;
    let (mx, my) = (
        hours.iter().sum::<f64>() / n,
        values.iter().sum::<f64>() / n,
    );
    let var: f64 = hours.iter().map(|x| (x - mx).powi(2)).sum();
    (var > 0.0).then(|| {
        hours
            .iter()
            .zip(values)
            .map(|(x, y)| (x - mx) * (y - my))
            .sum::<f64>()
            / var
    })
}

/// Compare the first and last third of the soak windows in `results`; steps
/// that are not soak windows (the warm-up) and windows without requests are
/// left out. `tolerance` is the share the last windows may be worse (0.2 is 20%).
pub fn analyze(results: &[BenchmarkResultsWriter], tolerance: f64) -> SoakSummary {
    let windows: Vec<&BenchmarkResultsWriter> = results
        .iter()
        .filter(|r| r.id.starts_with(WINDOW_PREFIX) && r.total_requests > 0)
        .collect();
    let window_secs = if windows.is_empty() {
        0.0
    } else {
        windows.iter().map(|w| w.duration_ms as f64).sum::<f64>() / windows.len() as f64 / 1000.0
    };
    let mut summary = SoakSummary {
        verdict: SoakVerdict::TooShort,
        windows: windows.len(),
        window_secs,
        tolerance,
        metrics: Vec::new(),
        reasons: Vec::new(),
    };
    if windows.len() < MIN_WINDOWS {
        summary.reasons.push(format!(
            "{} of the {MIN_WINDOWS} windows needed to see a trend had requests.",
            windows.len()
        ));
        return summary;
    }
    let third = (windows.len() / 3).max(1);
    let mut elapsed = 0.0;
    let hours: Vec<f64> = windows
        .iter()
        .map(|w| {
            elapsed += w.duration_ms as f64 / 3_600_000.0;
            elapsed
        })
        .collect();
    for m in &MEASURES {
        let series: Vec<f64> = windows.iter().map(|w| (m.read)(w)).collect();
        let (first, last) = (
            median(&series[..third]),
            median(&series[series.len() - third..]),
        );
        let change = (first > 0.0).then(|| (last - first) / first);
        let worse = match m.limit {
            Limit::Grew => change.is_some_and(|c| c > tolerance),
            Limit::Fell => change.is_some_and(|c| c < -tolerance),
            Limit::Rose => last - first > ERROR_RATE_RISE,
        };
        if worse {
            summary.reasons.push(match m.limit {
                Limit::Rose => format!(
                    "{} went from {:.1}% to {:.1}%.",
                    m.label,
                    first * 100.0,
                    last * 100.0
                ),
                _ => format!(
                    "{} went from {first:.1} {} to {last:.1} {} ({:+.0}%).",
                    m.label,
                    m.unit,
                    m.unit,
                    change.unwrap_or_default() * 100.0
                ),
            });
        }
        summary.metrics.push(DriftMetric {
            id: m.id.into(),
            label: m.label.into(),
            unit: m.unit.into(),
            first,
            last,
            change,
            slope_per_hour: slope(&series, &hours),
            worse,
        });
    }
    summary.verdict = if summary.metrics.iter().any(|m| m.worse) {
        SoakVerdict::Degrading
    } else {
        SoakVerdict::Stable
    };
    if summary.verdict == SoakVerdict::Stable {
        summary.reasons.push(format!(
            "The last {third} window(s) are within {:.0}% of the first {third}.",
            tolerance * 100.0
        ));
    }
    summary
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A window with the given p95 latency (ms), request rate and error rate.
    fn window(n: usize, e2e: f64, rps: f64, errors: f64) -> BenchmarkResultsWriter {
        let mut w = BenchmarkResultsWriter {
            id: format!("{WINDOW_PREFIX}{n:03}"),
            total_requests: 100,
            duration_ms: 300_000,
            request_rate: rps,
            error_rate: errors,
            ..Default::default()
        };
        w.e2e_latency_ms.p95 = e2e;
        w.time_to_first_token_ms.p95 = e2e / 10.0;
        w.inter_token_latency_ms.p95 = e2e / 100.0;
        w
    }

    fn series(f: impl Fn(usize) -> (f64, f64, f64), n: usize) -> Vec<BenchmarkResultsWriter> {
        (1..=n)
            .map(|i| {
                let (e, r, x) = f(i);
                window(i, e, r, x)
            })
            .collect()
    }

    #[test]
    fn a_flat_series_is_stable() {
        let s = analyze(
            &series(|i| (1000.0 + (i % 2) as f64 * 20.0, 10.0, 0.0), 12),
            DEFAULT_TOLERANCE,
        );
        assert_eq!((s.verdict, s.windows), (SoakVerdict::Stable, 12));
        assert!(s.metrics.iter().all(|m| !m.worse) && s.metrics.len() == 5);
        assert!((s.window_secs - 300.0).abs() < 1e-9);
    }

    #[test]
    fn growing_latency_is_degrading_and_says_by_how_much() {
        let s = analyze(
            &series(|i| (1000.0 + 100.0 * i as f64, 10.0, 0.0), 12),
            DEFAULT_TOLERANCE,
        );
        assert_eq!(s.verdict, SoakVerdict::Degrading);
        let e2e = s.metrics.iter().find(|m| m.id == "e2e_p95").unwrap();
        assert!(e2e.worse && e2e.change.unwrap() > 0.5, "{e2e:?}");
        assert!(e2e.slope_per_hour.unwrap() > 0.0);
        assert!(
            s.reasons[0].contains("Total time") && s.reasons[0].contains('%'),
            "{:?}",
            s.reasons
        );
    }

    #[test]
    fn falling_throughput_and_rising_errors_are_found() {
        let s = analyze(
            &series(|i| (1000.0, 10.0 - 0.5 * i as f64, 0.0), 10),
            DEFAULT_TOLERANCE,
        );
        assert_eq!(s.verdict, SoakVerdict::Degrading);
        assert!(s.metrics.iter().find(|m| m.id == "rps").unwrap().worse);
        let s = analyze(
            &series(|i| (1000.0, 10.0, if i > 6 { 0.05 } else { 0.0 }), 9),
            DEFAULT_TOLERANCE,
        );
        assert!(
            s.metrics
                .iter()
                .find(|m| m.id == "error_rate")
                .unwrap()
                .worse
        );
        assert!(s.reasons.iter().any(|r| r.contains("Failed requests")));
    }

    #[test]
    fn improvements_and_small_changes_are_not_worse() {
        let s = analyze(
            &series(|i| (2000.0 - 50.0 * i as f64, 10.0, 0.0), 12),
            DEFAULT_TOLERANCE,
        );
        assert_eq!(s.verdict, SoakVerdict::Stable);
        let s = analyze(
            &series(|i| (1000.0 + 5.0 * i as f64, 10.0, 0.0), 12),
            DEFAULT_TOLERANCE,
        );
        assert_eq!(s.verdict, SoakVerdict::Stable);
        // the same drift fails a tighter tolerance
        assert_eq!(
            analyze(&series(|i| (1000.0 + 5.0 * i as f64, 10.0, 0.0), 12), 0.02).verdict,
            SoakVerdict::Degrading
        );
    }

    #[test]
    fn warmup_and_empty_windows_are_ignored() {
        let mut results = series(|_| (1000.0, 10.0, 0.0), 6);
        results.insert(
            0,
            BenchmarkResultsWriter {
                id: "warmup".into(),
                total_requests: 5,
                ..Default::default()
            },
        );
        let mut empty = window(7, 0.0, 0.0, 0.0);
        empty.total_requests = 0;
        results.push(empty);
        let s = analyze(&results, DEFAULT_TOLERANCE);
        assert_eq!((s.verdict, s.windows), (SoakVerdict::Stable, 6));
    }

    #[test]
    fn two_windows_are_too_short() {
        let s = analyze(&series(|_| (1000.0, 10.0, 0.0), 2), DEFAULT_TOLERANCE);
        assert_eq!((s.verdict, s.windows), (SoakVerdict::TooShort, 2));
        assert!(s.metrics.is_empty());
        assert_eq!(
            analyze(&[], DEFAULT_TOLERANCE).verdict,
            SoakVerdict::TooShort
        );
    }

    #[test]
    fn median_and_slope() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 2.0, 3.0]), 2.5);
        assert!((slope(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]).unwrap() - 1.0).abs() < 1e-9);
        assert_eq!(slope(&[1.0, 2.0], &[1.0, 1.0]), None);
    }
}
