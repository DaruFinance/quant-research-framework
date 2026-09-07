//! Event-driven rebalance scheduler (HIGH-RISK Rust mirror).

#![cfg(feature = "carry")]

use crate::carry::funding::FundingFrame;
use crate::carry::triggers::TriggerEvent;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RebalanceKind {
    Bar,
    Funding,
    Trigger,
}

impl RebalanceKind {
    fn rank(self) -> u8 {
        match self {
            RebalanceKind::Bar => 0,
            RebalanceKind::Funding => 1,
            RebalanceKind::Trigger => 2,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ScheduledRebalance {
    pub time_s: i64,
    pub kind: RebalanceKind,
    pub tag: Option<String>,
    pub trigger_kind: Option<&'static str>,
}

pub struct EventDrivenScheduler<'a> {
    pub bar_cadence_s: Option<i64>,
    pub funding_frame: Option<&'a FundingFrame>,
    pub triggers: Vec<TriggerEvent>,
    pub t_start_s: i64,
    pub t_end_s: i64,
}

impl<'a> EventDrivenScheduler<'a> {
    pub fn new(
        bar_cadence_s: Option<i64>,
        funding_frame: Option<&'a FundingFrame>,
        triggers: Vec<TriggerEvent>,
        t_start_s: i64,
        t_end_s: Option<i64>,
    ) -> Self {
        Self {
            bar_cadence_s,
            funding_frame,
            triggers,
            t_start_s,
            // Mirror Python's 10**12 default cap.
            t_end_s: t_end_s.unwrap_or(1_000_000_000_000),
        }
    }

    pub fn run(&self) -> Vec<ScheduledRebalance> {
        let mut out = Vec::new();
        if let Some(cad) = self.bar_cadence_s {
            if cad > 0 {
                let mut t = self.t_start_s;
                while t <= self.t_end_s {
                    out.push(ScheduledRebalance {
                        time_s: t,
                        kind: RebalanceKind::Bar,
                        tag: None,
                        trigger_kind: None,
                    });
                    t += cad;
                }
            }
        }
        if let Some(frame) = self.funding_frame {
            for ev in &frame.events {
                if ev.time_s >= self.t_start_s && ev.time_s <= self.t_end_s {
                    out.push(ScheduledRebalance {
                        time_s: ev.time_s,
                        kind: RebalanceKind::Funding,
                        tag: Some("funding_settle".to_string()),
                        trigger_kind: None,
                    });
                }
            }
        }
        for ev in &self.triggers {
            if ev.time_s >= self.t_start_s && ev.time_s <= self.t_end_s {
                out.push(ScheduledRebalance {
                    time_s: ev.time_s,
                    kind: RebalanceKind::Trigger,
                    tag: Some(ev.kind.to_string()),
                    trigger_kind: Some(ev.kind),
                });
            }
        }
        out.sort_by(|a, b| {
            (a.time_s, a.kind.rank()).cmp(&(b.time_s, b.kind.rank()))
        });
        out
    }

    /// Next scheduled rebalance strictly after `after_s`.  Reads only
    /// `after_s` and the loaded streams; never consults future state
    /// it isn't supposed to.
    pub fn next_rebalance(&self, after_s: i64) -> Option<ScheduledRebalance> {
        let mut candidates: Vec<ScheduledRebalance> = Vec::new();
        if let Some(cad) = self.bar_cadence_s {
            if cad > 0 {
                let rem = (after_s - self.t_start_s).rem_euclid(cad);
                let nxt = if rem == 0 {
                    after_s + cad
                } else {
                    after_s + (cad - rem)
                };
                if nxt <= self.t_end_s {
                    candidates.push(ScheduledRebalance {
                        time_s: nxt,
                        kind: RebalanceKind::Bar,
                        tag: None,
                        trigger_kind: None,
                    });
                }
            }
        }
        if let Some(frame) = self.funding_frame {
            for ev in &frame.events {
                if ev.time_s > after_s {
                    if ev.time_s <= self.t_end_s {
                        candidates.push(ScheduledRebalance {
                            time_s: ev.time_s,
                            kind: RebalanceKind::Funding,
                            tag: Some("funding_settle".to_string()),
                            trigger_kind: None,
                        });
                    }
                    break;
                }
            }
        }
        for ev in &self.triggers {
            if ev.time_s > after_s && ev.time_s <= self.t_end_s {
                candidates.push(ScheduledRebalance {
                    time_s: ev.time_s,
                    kind: RebalanceKind::Trigger,
                    tag: Some(ev.kind.to_string()),
                    trigger_kind: Some(ev.kind),
                });
            }
        }
        candidates.sort_by(|a, b| {
            (a.time_s, a.kind.rank()).cmp(&(b.time_s, b.kind.rank()))
        });
        candidates.into_iter().next()
    }
}
