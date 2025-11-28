//! Composite druggability scoring

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringWeights {
    pub volume: f64,
    pub hydrophobicity: f64,
    pub enclosure: f64,
    pub depth: f64,
    pub hbond_capacity: f64,
    pub flexibility: f64,
    pub conservation: f64,
    pub topology: f64,
}

impl Default for ScoringWeights {
    fn default() -> Self {
        Self {
            volume: 0.15,
            hydrophobicity: 0.20,
            enclosure: 0.15,
            depth: 0.15,
            hbond_capacity: 0.10,
            flexibility: 0.05,
            conservation: 0.10,
            topology: 0.10,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DrugabilityClass {
    HighlyDruggable,
    Druggable,
    DifficultTarget,
    Undruggable,
}

impl Default for DrugabilityClass {
    fn default() -> Self {
        DrugabilityClass::Undruggable
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Components {
    pub volume: f64,
    pub hydro: f64,
    pub enclosure: f64,
    pub depth: f64,
    pub hbond: f64,
    pub flex: f64,
    pub cons: f64,
    pub topo: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DruggabilityScore {
    pub total: f64,
    pub classification: DrugabilityClass,
    pub components: Components,
}

pub struct DruggabilityScorer {
    weights: ScoringWeights,
}

impl DruggabilityScorer {
    pub fn new(weights: ScoringWeights) -> Self {
        Self { weights }
    }

    pub fn score(&self, pocket: &crate::pocket::Pocket) -> DruggabilityScore {
        let volume = self.score_volume(pocket.volume);
        let hydro = self.score_hydrophobicity(pocket.mean_hydrophobicity);
        let enclosure = self.score_enclosure(pocket.enclosure_ratio);
        let depth = self.score_depth(pocket.mean_depth);
        let hbond = self.score_hbond(pocket.hbond_donors, pocket.hbond_acceptors);
        let flex = self.score_flexibility(pocket.mean_flexibility);
        let cons = self.score_conservation(pocket.mean_conservation);
        let topo = pocket.persistence_score;

        let total = self.weights.volume * volume
            + self.weights.hydrophobicity * hydro
            + self.weights.enclosure * enclosure
            + self.weights.depth * depth
            + self.weights.hbond_capacity * hbond
            + self.weights.flexibility * flex
            + self.weights.conservation * cons
            + self.weights.topology * topo;

        DruggabilityScore {
            total,
            classification: self.classify(total),
            components: Components {
                volume,
                hydro,
                enclosure,
                depth,
                hbond,
                flex,
                cons,
                topo,
            },
        }
    }

    fn score_volume(&self, v: f64) -> f64 {
        // Sigmoid centered near 650 Å^3, clamped for extremely large pockets
        let x = (v - 650.0) / 250.0;
        1.0 / (1.0 + (-x).exp()).clamp(0.0, 1.0)
    }

    fn score_hydrophobicity(&self, h: f64) -> f64 {
        ((h + 4.5) / 9.0).clamp(0.0, 1.0)
    }

    fn score_enclosure(&self, e: f64) -> f64 {
        // prefer partially enclosed pockets; too enclosed can be penalized slightly
        let clamped = e.clamp(0.0, 1.0);
        if clamped < 0.2 {
            clamped * 0.5
        } else if clamped > 0.9 {
            0.9 - (clamped - 0.9) * 0.8
        } else {
            clamped
        }
    }

    fn score_depth(&self, d: f64) -> f64 {
        let normalized = (d / 12.0).clamp(0.0, 1.2);
        (1.0 / (1.0 + (-4.0 * (normalized - 0.5)).exp())).clamp(0.0, 1.0)
    }

    fn score_hbond(&self, donors: usize, acceptors: usize) -> f64 {
        let total = donors + acceptors;
        (total as f64 / 10.0).min(1.0)
    }

    fn score_flexibility(&self, flex: f64) -> f64 {
        (1.0 - (flex / 100.0)).clamp(0.0, 1.0)
    }

    fn score_conservation(&self, cons: f64) -> f64 {
        cons.clamp(0.0, 1.0)
    }

    fn classify(&self, score: f64) -> DrugabilityClass {
        if score >= 0.7 {
            DrugabilityClass::HighlyDruggable
        } else if score >= 0.5 {
            DrugabilityClass::Druggable
        } else if score >= 0.3 {
            DrugabilityClass::DifficultTarget
        } else {
            DrugabilityClass::Undruggable
        }
    }
}
