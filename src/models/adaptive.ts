/**
 * Adaptive Model Routing - choose model based on cost/speed/quality preference
 */

import { isLLMProviderId } from '../providers/llm-client';

export type ModelStrategy = 'cost' | 'speed' | 'quality' | 'balanced';

export interface AdaptiveModelInput {
  primary: string;
  fallbacks?: string[];
  strategy?: ModelStrategy;
}

interface ModelMeta {
  costScore: number; // lower cost => higher score
  speedScore: number;
  qualityScore: number;
}

// Heuristic metadata for known Anthropic models.
// Scores are relative (higher is better for the given dimension).
const MODEL_META: Record<string, ModelMeta> = {
  // Latest models
  'claude-opus-4-6': { costScore: 1, speedScore: 3, qualityScore: 10 },
  'claude-opus-4-5-20250514': { costScore: 2, speedScore: 4, qualityScore: 10 },
  'claude-sonnet-4-6': { costScore: 5, speedScore: 7, qualityScore: 9 },
  'claude-sonnet-4-5-20250929': { costScore: 5, speedScore: 7, qualityScore: 9 },
  'claude-sonnet-4-20250514': { costScore: 6, speedScore: 7, qualityScore: 8 },
  'claude-haiku-4-5-20251001': { costScore: 9, speedScore: 10, qualityScore: 7 },
  'claude-haiku-3-5-20250514': { costScore: 10, speedScore: 10, qualityScore: 6 },
  // Legacy IDs (kept for compatibility)
  'claude-3-5-haiku-20241022': { costScore: 10, speedScore: 10, qualityScore: 6 },
  'claude-3-5-sonnet-20241022': { costScore: 6, speedScore: 7, qualityScore: 8 },
  'claude-3-opus-20240229': { costScore: 2, speedScore: 4, qualityScore: 9 },
};

const DEFAULT_STRATEGY: ModelStrategy = 'quality';

function normalizeStrategy(value?: string): ModelStrategy {
  const v = (value || '').trim().toLowerCase();
  if (v === 'cost' || v === 'speed' || v === 'quality' || v === 'balanced') return v;
  return DEFAULT_STRATEGY;
}

function scoreFor(meta: ModelMeta, strategy: ModelStrategy): number {
  switch (strategy) {
    case 'cost':
      return meta.costScore;
    case 'speed':
      return meta.speedScore;
    case 'quality':
      return meta.qualityScore;
    case 'balanced':
    default:
      return meta.costScore * 0.3 + meta.speedScore * 0.3 + meta.qualityScore * 0.4;
  }
}

/**
 * Strip a known provider prefix (`anthropic/claude-...` → `claude-...`) while leaving
 * vendor-scoped model ids (`meta-llama/Llama-3.3-70B-Instruct-Turbo`) intact.
 */
export function stripProviderPrefix(model: string): string {
  const slash = model.indexOf('/');
  if (slash > 0 && isLLMProviderId(model.slice(0, slash))) {
    return model.slice(slash + 1) || model;
  }
  return model;
}

export function getModelStrategy(): ModelStrategy {
  return normalizeStrategy(process.env.AUTOPILOT_MODEL_STRATEGY);
}

export function selectAdaptiveModel(input: AdaptiveModelInput): string {
  const strategy = input.strategy || getModelStrategy();

  const candidates = [input.primary, ...(input.fallbacks || [])];

  // Score on the bare model id, but keep the original reference so the provider prefix
  // survives to the LLM client's routing layer.
  const scored = candidates
    .map((raw) => {
      const id = stripProviderPrefix(raw);
      return { raw, id, meta: MODEL_META[id] };
    })
    .filter((c) => Boolean(c.meta)) as Array<{ raw: string; id: string; meta: ModelMeta }>;

  if (scored.length === 0) {
    return input.primary;
  }

  scored.sort((a, b) => scoreFor(b.meta, strategy) - scoreFor(a.meta, strategy));
  return scored[0].raw;
}
