//! Model pricing tables for cost calculation.
//!
//! This module is application-layer only - the core library tracks tokens,
//! and the CLI applies pricing when displaying/exporting results.

use std::collections::HashMap;
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy)]
pub struct ModelPricing {
    pub input_per_1m: f64,
    pub output_per_1m: f64,
    pub cached_input_per_1m: Option<f64>,
}

impl ModelPricing {
    pub fn calculate_cost(&self, input_tokens: u64, output_tokens: u64, cached_tokens: u64) -> f64 {
        let input_cost = (input_tokens as f64 / 1_000_000.0) * self.input_per_1m;
        let output_cost = (output_tokens as f64 / 1_000_000.0) * self.output_per_1m;
        let cached_cost = if let Some(cached_rate) = self.cached_input_per_1m {
            (cached_tokens as f64 / 1_000_000.0) * cached_rate
        } else {
            0.0
        };
        input_cost + output_cost + cached_cost
    }
}

static PRICING_TABLE: OnceLock<HashMap<&'static str, ModelPricing>> = OnceLock::new();

fn init_pricing_table() -> HashMap<&'static str, ModelPricing> {
    let mut m = HashMap::new();

    // OpenAI GPT-4 family
    m.insert(
        "gpt-4o",
        ModelPricing {
            input_per_1m: 2.50,
            output_per_1m: 10.00,
            cached_input_per_1m: Some(1.25),
        },
    );
    m.insert(
        "gpt-4o-2024-11-20",
        ModelPricing {
            input_per_1m: 2.50,
            output_per_1m: 10.00,
            cached_input_per_1m: Some(1.25),
        },
    );
    m.insert(
        "gpt-4o-2024-08-06",
        ModelPricing {
            input_per_1m: 2.50,
            output_per_1m: 10.00,
            cached_input_per_1m: Some(1.25),
        },
    );
    m.insert(
        "gpt-4o-mini",
        ModelPricing {
            input_per_1m: 0.15,
            output_per_1m: 0.60,
            cached_input_per_1m: Some(0.075),
        },
    );
    m.insert(
        "gpt-4o-mini-2024-07-18",
        ModelPricing {
            input_per_1m: 0.15,
            output_per_1m: 0.60,
            cached_input_per_1m: Some(0.075),
        },
    );
    m.insert(
        "gpt-4-turbo",
        ModelPricing {
            input_per_1m: 10.00,
            output_per_1m: 30.00,
            cached_input_per_1m: None,
        },
    );
    m.insert(
        "gpt-4-turbo-preview",
        ModelPricing {
            input_per_1m: 10.00,
            output_per_1m: 30.00,
            cached_input_per_1m: None,
        },
    );
    m.insert(
        "gpt-4",
        ModelPricing {
            input_per_1m: 30.00,
            output_per_1m: 60.00,
            cached_input_per_1m: None,
        },
    );
    m.insert(
        "gpt-4-32k",
        ModelPricing {
            input_per_1m: 60.00,
            output_per_1m: 120.00,
            cached_input_per_1m: None,
        },
    );

    // OpenAI GPT-3.5 family
    m.insert(
        "gpt-3.5-turbo",
        ModelPricing {
            input_per_1m: 0.50,
            output_per_1m: 1.50,
            cached_input_per_1m: None,
        },
    );
    m.insert(
        "gpt-3.5-turbo-0125",
        ModelPricing {
            input_per_1m: 0.50,
            output_per_1m: 1.50,
            cached_input_per_1m: None,
        },
    );

    // OpenAI o1 family
    m.insert(
        "o1",
        ModelPricing {
            input_per_1m: 15.00,
            output_per_1m: 60.00,
            cached_input_per_1m: Some(7.50),
        },
    );
    m.insert(
        "o1-preview",
        ModelPricing {
            input_per_1m: 15.00,
            output_per_1m: 60.00,
            cached_input_per_1m: Some(7.50),
        },
    );
    m.insert(
        "o1-mini",
        ModelPricing {
            input_per_1m: 3.00,
            output_per_1m: 12.00,
            cached_input_per_1m: Some(1.50),
        },
    );
    m.insert(
        "o3-mini",
        ModelPricing {
            input_per_1m: 1.10,
            output_per_1m: 4.40,
            cached_input_per_1m: Some(0.55),
        },
    );

    // OpenAI embedding models
    m.insert(
        "text-embedding-3-small",
        ModelPricing {
            input_per_1m: 0.02,
            output_per_1m: 0.0,
            cached_input_per_1m: None,
        },
    );
    m.insert(
        "text-embedding-3-large",
        ModelPricing {
            input_per_1m: 0.13,
            output_per_1m: 0.0,
            cached_input_per_1m: None,
        },
    );
    m.insert(
        "text-embedding-ada-002",
        ModelPricing {
            input_per_1m: 0.10,
            output_per_1m: 0.0,
            cached_input_per_1m: None,
        },
    );

    // Anthropic Claude family
    m.insert(
        "claude-3-5-sonnet-20241022",
        ModelPricing {
            input_per_1m: 3.00,
            output_per_1m: 15.00,
            cached_input_per_1m: Some(0.30),
        },
    );
    m.insert(
        "claude-3-5-sonnet-latest",
        ModelPricing {
            input_per_1m: 3.00,
            output_per_1m: 15.00,
            cached_input_per_1m: Some(0.30),
        },
    );
    m.insert(
        "claude-sonnet-4-20250514",
        ModelPricing {
            input_per_1m: 3.00,
            output_per_1m: 15.00,
            cached_input_per_1m: Some(0.30),
        },
    );
    m.insert(
        "claude-3-5-haiku-20241022",
        ModelPricing {
            input_per_1m: 0.80,
            output_per_1m: 4.00,
            cached_input_per_1m: Some(0.08),
        },
    );
    m.insert(
        "claude-3-opus-20240229",
        ModelPricing {
            input_per_1m: 15.00,
            output_per_1m: 75.00,
            cached_input_per_1m: Some(1.50),
        },
    );
    m.insert(
        "claude-3-sonnet-20240229",
        ModelPricing {
            input_per_1m: 3.00,
            output_per_1m: 15.00,
            cached_input_per_1m: Some(0.30),
        },
    );
    m.insert(
        "claude-3-haiku-20240307",
        ModelPricing {
            input_per_1m: 0.25,
            output_per_1m: 1.25,
            cached_input_per_1m: Some(0.03),
        },
    );

    // Google Gemini family
    m.insert(
        "gemini-1.5-pro",
        ModelPricing {
            input_per_1m: 1.25,
            output_per_1m: 5.00,
            cached_input_per_1m: Some(0.3125),
        },
    );
    m.insert(
        "gemini-1.5-flash",
        ModelPricing {
            input_per_1m: 0.075,
            output_per_1m: 0.30,
            cached_input_per_1m: Some(0.01875),
        },
    );
    m.insert(
        "gemini-2.0-flash",
        ModelPricing {
            input_per_1m: 0.10,
            output_per_1m: 0.40,
            cached_input_per_1m: Some(0.025),
        },
    );
    m.insert(
        "gemini-2.0-flash-exp",
        ModelPricing {
            input_per_1m: 0.0,
            output_per_1m: 0.0,
            cached_input_per_1m: None,
        },
    );

    // Mistral family
    m.insert(
        "mistral-large-latest",
        ModelPricing {
            input_per_1m: 2.00,
            output_per_1m: 6.00,
            cached_input_per_1m: None,
        },
    );
    m.insert(
        "mistral-small-latest",
        ModelPricing {
            input_per_1m: 0.20,
            output_per_1m: 0.60,
            cached_input_per_1m: None,
        },
    );
    m.insert(
        "codestral-latest",
        ModelPricing {
            input_per_1m: 0.30,
            output_per_1m: 0.90,
            cached_input_per_1m: None,
        },
    );

    // DeepSeek
    m.insert(
        "deepseek-chat",
        ModelPricing {
            input_per_1m: 0.14,
            output_per_1m: 0.28,
            cached_input_per_1m: Some(0.014),
        },
    );
    m.insert(
        "deepseek-reasoner",
        ModelPricing {
            input_per_1m: 0.55,
            output_per_1m: 2.19,
            cached_input_per_1m: Some(0.14),
        },
    );

    m
}

pub fn get_pricing(model: &str) -> Option<ModelPricing> {
    let table = PRICING_TABLE.get_or_init(init_pricing_table);

    // Try exact match first
    if let Some(p) = table.get(model) {
        return Some(*p);
    }

    // Try prefix matching for versioned models
    for (key, pricing) in table.iter() {
        if model.starts_with(key) || key.starts_with(model) {
            return Some(*pricing);
        }
    }

    None
}

/// Calculate cost for a model given token counts.
/// Returns 0.0 if the model is not found in the pricing table.
pub fn calculate_cost(
    model: &str,
    input_tokens: u64,
    output_tokens: u64,
    cached_tokens: u64,
) -> f64 {
    match get_pricing(model) {
        Some(pricing) => pricing.calculate_cost(input_tokens, output_tokens, cached_tokens),
        None => 0.0,
    }
}

/// Calculate cost from aggregated metrics.
/// Uses gpt-4o-mini as default model for estimation when model is unknown.
pub fn calculate_cost_from_metrics(
    input_tokens: u64,
    output_tokens: u64,
    cached_tokens: u64,
) -> f64 {
    // Use gpt-4o-mini pricing as a reasonable default
    calculate_cost("gpt-4o-mini", input_tokens, output_tokens, cached_tokens)
}

/// Cost breakdown calculated from TokenStats.
#[derive(Debug, Clone, Default)]
pub struct CostBreakdown {
    pub agent_cost_usd: f64,
    pub judge_cost_usd: f64,
    pub total_cost_usd: f64,
}

/// Calculate cost from TokenStats using actual model info.
///
/// Uses the models recorded in TokenStats.models_used for accurate pricing.
/// Falls back to gpt-4o-mini if no model info is available.
pub fn calculate_cost_from_tokens(tokens: &pacabench_core::stats::TokenStats) -> CostBreakdown {
    // If we have per-model breakdown, price each model separately
    let mut agent_cost = 0.0;
    let mut judge_cost = 0.0;

    if !tokens.per_model.is_empty() {
        for (model, usage) in &tokens.per_model {
            agent_cost += calculate_cost(
                model,
                usage.agent_input_tokens,
                usage.agent_output_tokens,
                usage.agent_cached_tokens,
            );
            judge_cost += calculate_cost(
                model,
                usage.judge_input_tokens,
                usage.judge_output_tokens,
                usage.judge_cached_tokens,
            );
        }
    } else {
        // Use the first model found, or default to gpt-4o-mini
        let model = tokens
            .models_used
            .first()
            .map(|s| s.as_str())
            .unwrap_or("gpt-4o-mini");

        agent_cost = calculate_cost(
            model,
            tokens.agent_input_tokens,
            tokens.agent_output_tokens,
            tokens.agent_cached_tokens,
        );

        judge_cost = calculate_cost(
            model,
            tokens.judge_input_tokens,
            tokens.judge_output_tokens,
            tokens.judge_cached_tokens,
        );
    }

    CostBreakdown {
        agent_cost_usd: agent_cost,
        judge_cost_usd: judge_cost,
        total_cost_usd: agent_cost + judge_cost,
    }
}
