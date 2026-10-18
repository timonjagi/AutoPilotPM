/**
 * Provider-agnostic LLM client.
 *
 * The agent loop speaks the Anthropic Messages API shape: content blocks, tool_use /
 * tool_result, stop_reason, and a streaming surface that emits `text` events. Rewriting
 * that loop per vendor would be risky, so this module keeps Anthropic as the *internal*
 * wire format and translates at the edge:
 *
 *   anthropic/...                → @anthropic-ai/sdk (unchanged, native path)
 *   openai/groq/together/fireworks/ollama → OpenAI Chat Completions (OpenAI-compatible)
 *   gemini                       → Google Generative Language API (generateContent)
 *
 * Routing is decided per request from the model reference:
 *   1. explicit prefix           `gemini/gemini-2.5-flash`
 *   2. AUTOPILOT_LLM_PROVIDER    forces a provider when the model id is unprefixed
 *   3. model-name inference      claude-* → anthropic, gemini-* → gemini, gpt-* → openai
 *   4. first configured provider (oss / local model ids land here)
 *
 * When a route is chosen by inference but that provider has no credentials configured,
 * the request falls back to the first *available* provider and its default model instead
 * of failing — so a Claude-named helper model still works on a Gemini-only setup.
 */

import Anthropic from '@anthropic-ai/sdk';
import { randomUUID } from 'crypto';
import { logger } from '../utils/logger';

// =============================================================================
// TYPES
// =============================================================================

export type LLMProviderId =
  | 'anthropic'
  | 'openai'
  | 'gemini'
  | 'groq'
  | 'together'
  | 'fireworks'
  | 'ollama';

export type AnthropicParams = Anthropic.MessageCreateParamsNonStreaming;
export type AnthropicMessage = Anthropic.Message;
export type AnthropicContentBlock = Anthropic.ContentBlockParam;
export type AnthropicMessageParam = Anthropic.MessageParam;

export interface LLMRequestOptions {
  signal?: AbortSignal;
}

/** Streaming surface consumed by the agent (only `text` is used today). */
export interface LLMMessageStream {
  on(event: 'text', listener: (delta: string, fullText: string) => void): this;
  finalMessage(): Promise<AnthropicMessage>;
}

export interface LLMClient {
  messages: {
    create(params: AnthropicParams, options?: LLMRequestOptions): Promise<AnthropicMessage>;
    stream(params: AnthropicParams): LLMMessageStream;
  };
}

export interface LLMRoute {
  provider: LLMProviderId;
  model: string;
  /** True when the model reference carried an explicit `provider/model` prefix. */
  explicit: boolean;
}

/** Raised when the resolved provider has no credentials configured. */
export class MissingProviderKeyError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'MissingProviderKeyError';
  }
}

/** HTTP failure from a provider backend. `status` lets the retry policy classify it. */
export class ProviderRequestError extends Error {
  readonly status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = 'ProviderRequestError';
    this.status = status;
  }
}

// =============================================================================
// PROVIDER REGISTRY
// =============================================================================

interface ProviderSpec {
  /** Primary credential env var (absent → keyless provider). */
  keyEnv?: string;
  /** Fallback credential env var (e.g. Google accepts either key). */
  altKeyEnv?: string;
  baseUrl: string;
  baseUrlEnv?: string;
  modelEnv?: string;
  defaultModel: string;
  /** Where to obtain a key — used in error text. */
  hint: string;
}

const PROVIDER_SPECS: Record<LLMProviderId, ProviderSpec> = {
  anthropic: {
    keyEnv: 'ANTHROPIC_API_KEY',
    baseUrl: 'https://api.anthropic.com/v1',
    modelEnv: 'AUTOPILOT_ANTHROPIC_MODEL',
    defaultModel: 'claude-opus-4-6',
    hint: 'https://console.anthropic.com',
  },
  openai: {
    keyEnv: 'OPENAI_API_KEY',
    baseUrlEnv: 'OPENAI_BASE_URL',
    baseUrl: 'https://api.openai.com/v1',
    modelEnv: 'AUTOPILOT_OPENAI_MODEL',
    defaultModel: 'gpt-4o',
    hint: 'https://platform.openai.com/api-keys',
  },
  gemini: {
    keyEnv: 'GEMINI_API_KEY',
    altKeyEnv: 'GOOGLE_API_KEY',
    baseUrlEnv: 'GEMINI_BASE_URL',
    baseUrl: 'https://generativelanguage.googleapis.com/v1beta',
    modelEnv: 'AUTOPILOT_GEMINI_MODEL',
    defaultModel: 'gemini-2.5-flash',
    hint: 'https://aistudio.google.com/apikey',
  },
  groq: {
    keyEnv: 'GROQ_API_KEY',
    baseUrlEnv: 'GROQ_BASE_URL',
    baseUrl: 'https://api.groq.com/openai/v1',
    modelEnv: 'AUTOPILOT_GROQ_MODEL',
    defaultModel: 'llama-3.3-70b-versatile',
    hint: 'https://console.groq.com/keys',
  },
  together: {
    keyEnv: 'TOGETHER_API_KEY',
    baseUrlEnv: 'TOGETHER_BASE_URL',
    baseUrl: 'https://api.together.xyz/v1',
    modelEnv: 'AUTOPILOT_TOGETHER_MODEL',
    defaultModel: 'meta-llama/Llama-3.3-70B-Instruct-Turbo',
    hint: 'https://api.together.ai',
  },
  fireworks: {
    keyEnv: 'FIREWORKS_API_KEY',
    baseUrlEnv: 'FIREWORKS_BASE_URL',
    baseUrl: 'https://api.fireworks.ai/inference/v1',
    modelEnv: 'AUTOPILOT_FIREWORKS_MODEL',
    defaultModel: 'accounts/fireworks/models/llama-v3p3-70b-instruct',
    hint: 'https://fireworks.ai/api-keys',
  },
  ollama: {
    baseUrlEnv: 'OLLAMA_URL',
    baseUrl: 'http://localhost:11434/v1',
    modelEnv: 'AUTOPILOT_OLLAMA_MODEL',
    defaultModel: 'llama3',
    hint: 'https://ollama.com',
  },
};

/** Providers that can host arbitrary (unrecognised) model ids such as `llama3`. */
const OSS_PROVIDERS: LLMProviderId[] = ['groq', 'together', 'fireworks', 'ollama'];
const ALL_PROVIDERS = Object.keys(PROVIDER_SPECS) as LLMProviderId[];

// =============================================================================
// RESOLUTION HELPERS
// =============================================================================

function isProviderId(value: string): value is LLMProviderId {
  return (ALL_PROVIDERS as string[]).includes(value);
}

/** Whether `value` names a provider this client knows how to route to. */
export function isLLMProviderId(value: string): boolean {
  return isProviderId((value || '').trim().toLowerCase());
}

function normalizeProviderId(value?: string | null): LLMProviderId | null {
  const raw = (value || '').trim().toLowerCase();
  return isProviderId(raw) ? raw : null;
}

function providerApiKey(spec: ProviderSpec, env: NodeJS.ProcessEnv): string {
  const primary = spec.keyEnv ? (env[spec.keyEnv] || '').trim() : '';
  if (primary) return primary;
  const alt = spec.altKeyEnv ? (env[spec.altKeyEnv] || '').trim() : '';
  return alt;
}

export function isProviderAvailable(provider: LLMProviderId, env: NodeJS.ProcessEnv = process.env): boolean {
  const spec = PROVIDER_SPECS[provider];
  if (!spec.keyEnv) {
    // Keyless providers (ollama) are only "configured" when pointed at explicitly.
    return Boolean((env.OLLAMA_URL || '').trim());
  }
  return Boolean(providerApiKey(spec, env));
}

export function listAvailableLLMProviders(env: NodeJS.ProcessEnv = process.env): LLMProviderId[] {
  const forced = normalizeProviderId(env.AUTOPILOT_LLM_PROVIDER);
  const rest = ALL_PROVIDERS.filter((p) => p !== forced && isProviderAvailable(p, env));
  return forced && isProviderAvailable(forced, env) ? [forced, ...rest] : rest;
}

export function defaultModelFor(provider: LLMProviderId, env: NodeJS.ProcessEnv = process.env): string {
  const spec = PROVIDER_SPECS[provider];
  const override = spec.modelEnv ? (env[spec.modelEnv] || '').trim() : '';
  return override || spec.defaultModel;
}

export function providerBaseUrl(provider: LLMProviderId, env: NodeJS.ProcessEnv = process.env): string {
  const spec = PROVIDER_SPECS[provider];
  const override = spec.baseUrlEnv ? (env[spec.baseUrlEnv] || '').trim() : '';
  const base = override || spec.baseUrl;
  if (provider === 'ollama') {
    // OLLAMA_URL is the server root (http://host:11434) unless it already targets /v1.
    return /\/v\d+(?:\/|$)/.test(base) ? base.replace(/\/$/, '') : `${base.replace(/\/$/, '')}/v1`;
  }
  return base.replace(/\/$/, '');
}

function missingKeyMessage(provider: LLMProviderId, env: NodeJS.ProcessEnv): string {
  const spec = PROVIDER_SPECS[provider];
  if (!spec.keyEnv) {
    return `Provider '${provider}' is not configured: set OLLAMA_URL (e.g. http://localhost:11434) in .env.`;
  }
  const alts = spec.altKeyEnv ? ` or ${spec.altKeyEnv}` : '';
  return `Provider '${provider}' is not configured: set ${spec.keyEnv}${alts} in .env (get a key at ${spec.hint}).`;
}

function noProviderConfiguredMessage(): string {
  const names = ALL_PROVIDERS.map((p) => PROVIDER_SPECS[p].keyEnv).filter(Boolean).join(', ');
  return (
    `No LLM provider configured. Set one of: ${names} — or point OLLAMA_URL at a local Ollama server. ` +
    `Use AUTOPILOT_LLM_PROVIDER to pick the default when several are configured.`
  );
}

/**
 * Recognise which provider a bare model id belongs to. Returns null for ids that
 * carry no provider signature (oss model names, vendor-scoped names like
 * `meta-llama/Llama-3.3-70B-Instruct-Turbo`).
 */
export function inferProviderFromModel(model: string): LLMProviderId | null {
  const m = model.trim().toLowerCase();
  if (!m) return null;
  if (m.startsWith('claude')) return 'anthropic';
  if (m.startsWith('gemini')) return 'gemini';
  if (/^(gpt-|chatgpt-|text-embedding-|codex-|o\d+(?:-|\d|$))/.test(m)) return 'openai';
  return null;
}

/** Can `provider` legitimately serve `model`? */
function fitsProvider(provider: LLMProviderId, model: string): boolean {
  const inferred = inferProviderFromModel(model);
  if (inferred) return inferred === provider;
  // No signature: only the open-compatible providers accept arbitrary model ids.
  return OSS_PROVIDERS.includes(provider);
}

function parseExplicitPrefix(
  raw: string,
): { provider: LLMProviderId; rest: string } | null {
  const slash = raw.indexOf('/');
  if (slash <= 0) return null;
  const head = raw.slice(0, slash).toLowerCase();
  if (!isProviderId(head)) return null;
  return { provider: head, rest: raw.slice(slash + 1).trim() };
}

/**
 * Resolve a model reference to a concrete `{provider, model}` pair.
 * Throws MissingProviderKeyError when nothing is usable.
 */
export function resolveLLMRoute(
  modelRef: string | undefined,
  env: NodeJS.ProcessEnv = process.env,
): LLMRoute {
  let raw = (modelRef || '').trim();

  // 1. Explicit `provider/model` prefix wins over everything.
  const explicit = parseExplicitPrefix(raw);
  if (explicit) {
    if (isProviderAvailable(explicit.provider, env)) {
      const model =
        explicit.rest && fitsProvider(explicit.provider, explicit.rest)
          ? explicit.rest
          : defaultModelFor(explicit.provider, env);
      return { provider: explicit.provider, model, explicit: true };
    }
    // The named provider has no credentials. Fall back to whatever is configured
    // rather than hard-failing: `anthropic/claude-opus-4-6` is the *shipped default*
    // primary, so a user who only adds GEMINI_API_KEY must not hit a wall on their
    // first message. Still fail loudly when nothing at all is configured.
    if (listAvailableLLMProviders(env).length === 0) {
      throw new MissingProviderKeyError(missingKeyMessage(explicit.provider, env));
    }
    logger.warn(
      { requested: raw, unavailableProvider: explicit.provider },
      'Requested provider has no credentials — falling back to a configured provider',
    );
    raw = explicit.rest || raw;
  }

  const forced = normalizeProviderId(env.AUTOPILOT_LLM_PROVIDER);
  if (forced && !isProviderAvailable(forced, env)) {
    throw new MissingProviderKeyError(missingKeyMessage(forced, env));
  }

  const candidates = forced ? [forced] : listAvailableLLMProviders(env);

  // 2. Prefer a configured provider that can actually serve this model id.
  if (raw) {
    const fitting = candidates.find((p) => fitsProvider(p, raw));
    if (fitting) return { provider: fitting, model: raw, explicit: false };
  }

  // 3. Nothing fits (model belongs elsewhere, or no signature at all) → provider default.
  const fallback = candidates[0];
  if (!fallback) throw new MissingProviderKeyError(noProviderConfiguredMessage());
  if (!isProviderAvailable(fallback, env)) {
    throw new MissingProviderKeyError(missingKeyMessage(fallback, env));
  }
  if (raw) {
    logger.debug(
      { requestedModel: raw, provider: fallback },
      'Model not served by configured providers — using provider default',
    );
  }
  return { provider: fallback, model: defaultModelFor(fallback, env), explicit: false };
}

/**
 * Whether `/model <ref>` should accept a model reference. Mirrors routing rules so the
 * command rejects only genuinely unusable values.
 */
export function isModelRefSupported(ref: string, env: NodeJS.ProcessEnv = process.env): boolean {
  const raw = (ref || '').trim();
  if (!raw) return false;
  try {
    resolveLLMRoute(raw, env);
    return true;
  } catch (err) {
    if (err instanceof MissingProviderKeyError) {
      // A known provider prefix / model signature is a valid *reference* even when its
      // key is absent — report the value to the user rather than silently re-routing.
      return Boolean(parseExplicitPrefix(raw) || inferProviderFromModel(raw));
    }
    return false;
  }
}

/**
 * Startup-level configuration check. Returns null when at least one provider is usable,
 * otherwise a message describing exactly what to fix.
 */
export function describeLLMConfigProblem(env: NodeJS.ProcessEnv = process.env): string | null {
  const rawForced = (env.AUTOPILOT_LLM_PROVIDER || '').trim();
  if (rawForced) {
    const forced = normalizeProviderId(rawForced);
    if (!forced) {
      return (
        `AUTOPILOT_LLM_PROVIDER='${rawForced}' is not a known provider. ` +
        `Expected one of: ${ALL_PROVIDERS.join(', ')}.`
      );
    }
    if (!isProviderAvailable(forced, env)) return missingKeyMessage(forced, env);
  }
  if (listAvailableLLMProviders(env).length === 0) return noProviderConfiguredMessage();
  return null;
}

// =============================================================================
// ANTHROPIC SHAPE HELPERS
// =============================================================================

function systemToString(system: AnthropicParams['system']): string {
  if (!system) return '';
  if (typeof system === 'string') return system;
  return system
    .filter((block) => block.type === 'text')
    .map((block) => (block as { text: string }).text)
    .join('\n\n');
}

function blocksOf(message: AnthropicMessageParam): AnthropicContentBlock[] {
  if (typeof message.content === 'string') {
    return [{ type: 'text', text: message.content }];
  }
  return message.content as AnthropicContentBlock[];
}

type AnyBlock = Record<string, unknown> & { type?: string };

function toolResultToText(content: unknown): string {
  if (content === undefined || content === null) return '';
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const parts: string[] = [];
    for (const block of content as AnyBlock[]) {
      if (block?.type === 'text' && typeof block.text === 'string') parts.push(block.text);
      else if (block?.type === 'image') parts.push('[image]');
      else if (block) parts.push(JSON.stringify(block));
    }
    return parts.filter(Boolean).join('\n');
  }
  if (typeof content === 'object') {
    try {
      return JSON.stringify(content);
    } catch {
      return String(content);
    }
  }
  return String(content);
}

function safeParseJSONObject(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return {};
  }
}

/** tool_use_id → tool name, gathered from assistant turns already in the history. */
function collectToolNameById(params: AnthropicParams): Map<string, string> {
  const map = new Map<string, string>();
  for (const message of params.messages || []) {
    for (const block of blocksOf(message) as unknown as AnyBlock[]) {
      if (block?.type === 'tool_use' && typeof block.id === 'string' && typeof block.name === 'string') {
        map.set(block.id, block.name);
      }
    }
  }
  return map;
}

function makeMessage(args: {
  id?: string;
  model: string;
  content: unknown[];
  stop_reason: Anthropic.StopReason | null;
  input_tokens: number;
  output_tokens: number;
  cache_read_input_tokens?: number;
  cache_creation_input_tokens?: number;
}): AnthropicMessage {
  return {
    id: args.id || `msg_${randomUUID().replace(/-/g, '')}`,
    container: null,
    content: args.content as Anthropic.ContentBlock[],
    model: args.model,
    role: 'assistant',
    stop_reason: args.stop_reason,
    stop_sequence: null,
    type: 'message',
    usage: {
      cache_creation: null,
      cache_creation_input_tokens: args.cache_creation_input_tokens ?? null,
      cache_read_input_tokens: args.cache_read_input_tokens ?? null,
      input_tokens: args.input_tokens,
      output_tokens: args.output_tokens,
    },
  } as unknown as AnthropicMessage;
}

// =============================================================================
// OPENAI-COMPATIBLE BACKEND (chat/completions)
// =============================================================================

export interface OpenAIChatBody {
  model: string;
  messages: Array<Record<string, unknown>>;
  tools?: Array<Record<string, unknown>>;
  tool_choice?: unknown;
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
  stop?: string | string[];
  stream?: boolean;
  stream_options?: { include_usage: boolean };
}

function toOpenAIContentPart(block: AnyBlock): Record<string, unknown> | null {
  if (block.type === 'text' && typeof block.text === 'string') {
    return { type: 'text', text: block.text };
  }
  if (block.type === 'image') {
    const source = block.source as { type?: string; media_type?: string; data?: string; url?: string } | undefined;
    if (source?.type === 'base64' && source.data && source.media_type) {
      return { type: 'image_url', image_url: { url: `data:${source.media_type};base64,${source.data}` } };
    }
    if (source?.type === 'url' && source.url) {
      return { type: 'image_url', image_url: { url: source.url } };
    }
  }
  return null;
}

export function toOpenAIMessages(params: AnthropicParams): Array<Record<string, unknown>> {
  const out: Array<Record<string, unknown>> = [];
  const system = systemToString(params.system);
  if (system) out.push({ role: 'system', content: system });

  for (const message of params.messages || []) {
    const blocks = blocksOf(message) as unknown as AnyBlock[];

    if (message.role === 'assistant') {
      const text = blocks
        .filter((b) => b.type === 'text' && typeof b.text === 'string')
        .map((b) => b.text as string)
        .join('');
      const toolUses = blocks.filter((b) => b.type === 'tool_use');
      if (!text && toolUses.length === 0) continue; // nothing OpenAI can represent (e.g. thinking only)
      const entry: Record<string, unknown> = { role: 'assistant', content: text || null };
      if (toolUses.length) {
        entry.tool_calls = toolUses.map((b) => ({
          id: String(b.id || `call_${randomUUID().replace(/-/g, '')}`),
          type: 'function',
          function: {
            name: String(b.name),
            arguments: typeof b.input === 'string' ? b.input : JSON.stringify(b.input ?? {}),
          },
        }));
      }
      out.push(entry);
      continue;
    }

    // user turn (assistant turns are handled above and continue)
    const toolResults = blocks.filter((b) => b.type === 'tool_result');
    for (const result of toolResults) {
      out.push({
        role: 'tool',
        tool_call_id: String(result.tool_use_id || ''),
        content: toolResultToText(result.content),
      });
    }
    const parts = blocks
      .filter((b) => b.type !== 'tool_result')
      .map((b) => toOpenAIContentPart(b))
      .filter((p): p is Record<string, unknown> => Boolean(p));
    if (parts.length === 0) continue;
    const allText = parts.every((p) => p.type === 'text');
    out.push({
      role: 'user',
      content: allText ? parts.map((p) => p.text).join('') : parts,
    });
  }

  return out;
}

export function toOpenAITools(params: AnthropicParams): Array<Record<string, unknown>> | undefined {
  const tools = (params.tools || []) as unknown as Array<Record<string, unknown>>;
  const out = tools
    .filter((tool) => typeof tool.name === 'string')
    .map((tool) => ({
      type: 'function',
      function: {
        name: tool.name,
        description: typeof tool.description === 'string' ? tool.description : '',
        parameters: tool.input_schema || { type: 'object', properties: {} },
      },
    }));
  return out.length ? out : undefined;
}

function toOpenAIToolChoice(params: AnthropicParams): unknown {
  const choice = (params as { tool_choice?: unknown }).tool_choice as
    | { type?: string; name?: string }
    | undefined;
  if (!choice?.type) return undefined;
  if (choice.type === 'none') return 'none';
  if (choice.type === 'any') return 'required';
  if (choice.type === 'tool' && choice.name) {
    return { type: 'function', function: { name: choice.name } };
  }
  if (choice.type === 'auto') return 'auto';
  return undefined;
}

/** Providers that reliably accept `stream_options.include_usage`. */
const STREAM_USAGE_PROVIDERS: LLMProviderId[] = ['openai', 'groq', 'together'];

export function buildOpenAIChatBody(
  params: AnthropicParams,
  model: string,
  stream: boolean,
  provider: LLMProviderId,
): OpenAIChatBody {
  const body: OpenAIChatBody = {
    model,
    messages: toOpenAIMessages(params),
    stream,
  };
  const tools = toOpenAITools(params);
  if (tools) body.tools = tools;
  const toolChoice = toOpenAIToolChoice(params);
  if (toolChoice !== undefined) body.tool_choice = toolChoice;
  if (typeof params.temperature === 'number') body.temperature = params.temperature;
  if (typeof params.top_p === 'number') body.top_p = params.top_p;
  if (typeof params.max_tokens === 'number') body.max_tokens = params.max_tokens;
  const stop = (params as { stop_sequences?: string[] }).stop_sequences;
  if (Array.isArray(stop) && stop.length) body.stop = stop;
  if (stream && STREAM_USAGE_PROVIDERS.includes(provider)) {
    body.stream_options = { include_usage: true };
  }
  return body;
}

function openAIFinishToStopReason(finish: string | undefined, hasToolUse: boolean): Anthropic.StopReason {
  if (hasToolUse || finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'length') return 'max_tokens';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

function toolCallBlocks(toolCalls: Array<Record<string, unknown>>): unknown[] {
  return toolCalls.map((call) => {
    const fn = (call.function || {}) as { name?: string; arguments?: string };
    const args = typeof fn.arguments === 'string' && fn.arguments ? safeParseJSONObject(fn.arguments) : {};
    return {
      type: 'tool_use',
      id: typeof call.id === 'string' && call.id ? call.id : `toolu_${randomUUID().replace(/-/g, '')}`,
      name: fn.name || '',
      input: args,
    };
  });
}

export function messageFromOpenAI(json: any, model: string): AnthropicMessage {
  const choice = Array.isArray(json?.choices) ? json.choices[0] : undefined;
  const message = choice?.message || {};
  const blocks: unknown[] = [];
  if (typeof message.content === 'string' && message.content) {
    blocks.push({ type: 'text', text: message.content });
  }
  if (Array.isArray(message.tool_calls) && message.tool_calls.length) {
    blocks.push(...toolCallBlocks(message.tool_calls));
  }
  if (blocks.length === 0) blocks.push({ type: 'text', text: '' });

  const usage = json?.usage || {};
  return makeMessage({
    id: typeof json?.id === 'string' ? json.id : undefined,
    model: typeof json?.model === 'string' && json.model ? json.model : model,
    content: blocks,
    stop_reason: openAIFinishToStopReason(choice?.finish_reason, blocks.some((b) => (b as { type: string }).type === 'tool_use')),
    input_tokens: usage.prompt_tokens ?? 0,
    output_tokens: usage.completion_tokens ?? 0,
    cache_read_input_tokens: usage.prompt_tokens_details?.cached_tokens ?? undefined,
  });
}

/** Incrementally folds OpenAI streaming chunks into an Anthropic-shaped message. */
export class OpenAIStreamAccumulator {
  private textParts: string[] = [];
  private toolCalls = new Map<number, { id?: string; name: string; args: string }>();
  private finishReason?: string;
  private inputTokens = 0;
  private outputTokens = 0;
  private cacheRead = 0;

  /** Consume one SSE payload; returns any text delta it carried. */
  push(chunk: any): string {
    const choice = Array.isArray(chunk?.choices) ? chunk.choices[0] : undefined;
    const delta = choice?.delta || {};
    let deltaText = '';

    if (typeof delta.content === 'string' && delta.content) {
      this.textParts.push(delta.content);
      deltaText = delta.content;
    }

    if (Array.isArray(delta.tool_calls)) {
      for (const call of delta.tool_calls) {
        const index = typeof call.index === 'number' ? call.index : 0;
        const current = this.toolCalls.get(index) || { name: '', args: '' };
        const fn = call.function || {};
        if (typeof call.id === 'string' && call.id) current.id = call.id;
        if (typeof fn.name === 'string' && fn.name) current.name += fn.name;
        if (typeof fn.arguments === 'string' && fn.arguments) current.args += fn.arguments;
        this.toolCalls.set(index, current);
      }
    }

    if (typeof choice?.finish_reason === 'string' && choice.finish_reason) {
      this.finishReason = choice.finish_reason;
    }

    const usage = chunk?.usage;
    if (usage) {
      this.inputTokens = usage.prompt_tokens ?? this.inputTokens;
      this.outputTokens = usage.completion_tokens ?? this.outputTokens;
      this.cacheRead = usage.prompt_tokens_details?.cached_tokens ?? this.cacheRead;
    }

    return deltaText;
  }

  get fullText(): string {
    return this.textParts.join('');
  }

  finish(model: string): AnthropicMessage {
    const blocks: unknown[] = [];
    const text = this.fullText;
    if (text) blocks.push({ type: 'text', text });
    if (this.toolCalls.size) {
      blocks.push(
        ...toolCallBlocks(
          [...this.toolCalls.entries()]
            .sort((a, b) => a[0] - b[0])
            .map(([, call]) => ({
              id: call.id,
              function: { name: call.name, arguments: call.args || '{}' },
            })),
        ),
      );
    }
    if (blocks.length === 0) blocks.push({ type: 'text', text: '' });

    return makeMessage({
      model,
      content: blocks,
      stop_reason: openAIFinishToStopReason(this.finishReason, this.toolCalls.size > 0),
      input_tokens: this.inputTokens,
      output_tokens: this.outputTokens,
      cache_read_input_tokens: this.cacheRead || undefined,
    });
  }
}

// =============================================================================
// GEMINI BACKEND (generativelanguage.googleapis.com)
// =============================================================================

export interface GeminiRequestBody {
  contents: Array<{ role: 'user' | 'model'; parts: Array<Record<string, unknown>> }>;
  systemInstruction?: { parts: Array<{ text: string }> };
  tools?: Array<{ functionDeclarations: Array<Record<string, unknown>> }>;
  generationConfig?: Record<string, unknown>;
}

/** Gemini function names must match [a-zA-Z_][a-zA-Z0-9_-]* and stay under 64 chars. */
export function sanitizeGeminiToolName(name: string): string {
  let out = (name || '').replace(/[^a-zA-Z0-9_-]/g, '_');
  if (!/^[a-zA-Z_]/.test(out)) out = `_${out}`;
  return out.slice(0, 64);
}

/** Strip JSON-Schema keywords Gemini's function-declaration subset rejects. */
export function sanitizeGeminiSchema(schema: unknown): Record<string, unknown> | undefined {
  if (!schema || typeof schema !== 'object') return undefined;
  const drop = new Set(['$schema', '$id', 'additionalProperties', 'examples', 'title', 'default']);
  const walk = (value: unknown): unknown => {
    if (Array.isArray(value)) return value.map(walk);
    if (!value || typeof value !== 'object') return value;
    const out: Record<string, unknown> = {};
    for (const [key, child] of Object.entries(value as Record<string, unknown>)) {
      if (drop.has(key)) continue;
      // Gemini's schema subset only accepts string enum members and rejects a numeric
      // one with a 400 whatever the declared type is (verified live: `type: number` and
      // `type: string` both fail). Dropping the enum keeps the declared type intact so
      // the tool still receives a number instead of the "60" that stringifying would
      // produce — and these tools already spell the allowed values out in the
      // description, which is the part the model actually reads.
      if (key === 'enum' && Array.isArray(child) && child.some((v) => typeof v !== 'string')) continue;
      out[key] = walk(child);
    }
    return out;
  };
  return walk(schema) as Record<string, unknown>;
}

export function buildGeminiRequestBody(params: AnthropicParams): GeminiRequestBody {
  const toolNames = collectToolNameById(params);
  const contents: GeminiRequestBody['contents'] = [];

  const pushParts = (role: 'user' | 'model', parts: Array<Record<string, unknown>>): void => {
    if (parts.length === 0) return;
    // Gemini rejects consecutive same-role turns in some models — merge instead.
    const last = contents[contents.length - 1];
    if (last && last.role === role) {
      last.parts.push(...parts);
      return;
    }
    contents.push({ role, parts });
  };

  for (const message of params.messages || []) {
    const blocks = blocksOf(message) as unknown as AnyBlock[];
    const parts: Array<Record<string, unknown>> = [];

    for (const block of blocks) {
      if (block.type === 'text' && typeof block.text === 'string' && block.text) {
        parts.push({ text: block.text });
      } else if (block.type === 'tool_use') {
        parts.push({
          functionCall: {
            name: sanitizeGeminiToolName(String(block.name || 'tool')),
            args:
              typeof block.input === 'string'
                ? safeParseJSONObject(block.input)
                : (block.input as Record<string, unknown>) ?? {},
          },
          // Required by Gemini 3.x: a replayed function call without its signature is a
          // 400. The signature rides as a sibling of functionCall, exactly as Gemini
          // returned it. Calls synthesized by us (or by Anthropic) have none — omitting
          // is correct there, since there is no signature to lose.
          ...(typeof block.thought_signature === 'string' && block.thought_signature
            ? { thoughtSignature: block.thought_signature }
            : {}),
        });
      } else if (block.type === 'tool_result') {
        const originalName = toolNames.get(String(block.tool_use_id)) || 'unknown_tool';
        const raw = toolResultToText(block.content);
        let parsed: unknown = raw;
        try {
          parsed = JSON.parse(raw);
        } catch {
          parsed = raw;
        }
        parts.push({
          functionResponse: {
            name: sanitizeGeminiToolName(originalName),
            response: { output: parsed },
            ...(block.is_error ? { error: { message: raw } } : {}),
          },
        });
      } else if (block.type === 'image') {
        const source = block.source as { type?: string; media_type?: string; data?: string } | undefined;
        if (source?.type === 'base64' && source.data && source.media_type) {
          parts.push({ inlineData: { mimeType: source.media_type, data: source.data } });
        }
      }
    }

    pushParts(message.role === 'assistant' ? 'model' : 'user', parts);
  }

  const system = systemToString(params.system);
  const tools = (params.tools || []) as unknown as Array<Record<string, unknown>>;
  const declarations = tools
    .filter((tool) => typeof tool.name === 'string')
    .map((tool) => {
      const declaration: Record<string, unknown> = {
        name: sanitizeGeminiToolName(String(tool.name)),
        description: typeof tool.description === 'string' ? tool.description : '',
      };
      const schema = sanitizeGeminiSchema(tool.input_schema);
      if (schema) declaration.parameters = schema;
      return declaration;
    });

  const generationConfig: Record<string, unknown> = {};
  if (typeof params.max_tokens === 'number') generationConfig.maxOutputTokens = params.max_tokens;
  if (typeof params.temperature === 'number') generationConfig.temperature = params.temperature;
  if (typeof params.top_p === 'number') generationConfig.topP = params.top_p;
  const stop = (params as { stop_sequences?: string[] }).stop_sequences;
  if (Array.isArray(stop) && stop.length) generationConfig.stopSequences = stop;

  const body: GeminiRequestBody = { contents };
  if (system) body.systemInstruction = { parts: [{ text: system }] };
  if (declarations.length) body.tools = [{ functionDeclarations: declarations }];
  if (Object.keys(generationConfig).length) body.generationConfig = generationConfig;
  return body;
}

export function messageFromGemini(json: any, model: string, params?: AnthropicParams): AnthropicMessage {
  const candidate = Array.isArray(json?.candidates) ? json.candidates[0] : undefined;
  const parts: any[] = candidate?.content?.parts || [];

  const toolNames = params ? collectToolNameById(params) : new Map<string, string>();
  const declared = new Map<string, string>();
  for (const tool of (params?.tools || []) as unknown as Array<Record<string, unknown>>) {
    if (typeof tool.name === 'string') {
      declared.set(sanitizeGeminiToolName(String(tool.name)), String(tool.name));
    }
  }

  const blocks: unknown[] = [];
  for (const part of parts) {
    if (typeof part.text === 'string' && part.text) {
      blocks.push({ type: 'text', text: part.text });
    } else if (part.functionCall) {
      const wireName = String(part.functionCall.name || '');
      blocks.push({
        type: 'tool_use',
        id: `toolu_${randomUUID().replace(/-/g, '')}`,
        name: declared.get(wireName) || toolNames.get(wireName) || wireName,
        input: (part.functionCall.args as Record<string, unknown>) ?? {},
        // Gemini 3.x ships the signature as a *sibling* of functionCall and rejects a
        // replayed call that omits it (400 "missing a thought_signature"). Anthropic's
        // wire format has nowhere to put it, so park it on the block and send it back in
        // buildGeminiRequestBody. Text parts don't require one (verified live) and lose
        // fidelity when merged, so only calls round-trip a signature.
        ...(typeof part.thoughtSignature === 'string' && part.thoughtSignature
          ? { thought_signature: part.thoughtSignature }
          : {}),
      });
    }
  }
  if (blocks.length === 0) blocks.push({ type: 'text', text: '' });

  const hasToolUse = blocks.some((b) => (b as { type: string }).type === 'tool_use');
  const finish = candidate?.finishReason;
  const stopReason: Anthropic.StopReason = hasToolUse
    ? 'tool_use'
    : finish === 'MAX_TOKENS'
      ? 'max_tokens'
      : finish === 'SAFETY' || finish === 'RECITATION' || finish === 'BLOCKLIST'
        ? 'refusal'
        : 'end_turn';

  const usage = json?.usageMetadata || {};
  return makeMessage({
    model: typeof json?.modelVersion === 'string' && json.modelVersion ? json.modelVersion : model,
    content: blocks,
    stop_reason: stopReason,
    input_tokens: usage.promptTokenCount ?? 0,
    output_tokens: usage.candidatesTokenCount ?? usage.totalTokenCount ?? 0,
  });
}

/** Incrementally folds Gemini streaming chunks into an Anthropic-shaped message. */
export class GeminiStreamAccumulator {
  private textParts: string[] = [];
  private toolCalls: Array<{ name: string; args: Record<string, unknown>; signature?: string }> = [];
  /** Signature seen before its functionCall chunk arrived (chunks are unordered). */
  private pendingSignature?: string;
  private finishReason?: string;
  private inputTokens = 0;
  private outputTokens = 0;

  push(chunk: any): string {
    const candidate = Array.isArray(chunk?.candidates) ? chunk.candidates[0] : undefined;
    let delta = '';
    for (const part of candidate?.content?.parts || []) {
      if (typeof part.text === 'string' && part.text) {
        this.textParts.push(part.text);
        delta += part.text;
      } else if (part.functionCall) {
        const own =
          typeof part.thoughtSignature === 'string' && part.thoughtSignature
            ? part.thoughtSignature
            : this.pendingSignature;
        this.toolCalls.push({
          name: String(part.functionCall.name || ''),
          args: (part.functionCall.args as Record<string, unknown>) ?? {},
          ...(own ? { signature: own } : {}),
        });
        this.pendingSignature = undefined;
      } else if (typeof part.thoughtSignature === 'string' && part.thoughtSignature) {
        // Signature arrived as its own part: pair it with the most recent unsigned call,
        // or hold it until the call chunk shows up.
        const pending = [...this.toolCalls].reverse().find((c) => !c.signature);
        if (pending) pending.signature = part.thoughtSignature;
        else this.pendingSignature = part.thoughtSignature;
      }
    }
    if (typeof candidate?.finishReason === 'string' && candidate.finishReason) {
      this.finishReason = candidate.finishReason;
    }
    const usage = chunk?.usageMetadata;
    if (usage) {
      this.inputTokens = usage.promptTokenCount ?? this.inputTokens;
      this.outputTokens = usage.candidatesTokenCount ?? usage.totalTokenCount ?? this.outputTokens;
    }
    return delta;
  }

  get fullText(): string {
    return this.textParts.join('');
  }

  finish(model: string, params?: AnthropicParams): AnthropicMessage {
    const declared = new Map<string, string>();
    for (const tool of (params?.tools || []) as unknown as Array<Record<string, unknown>>) {
      if (typeof tool.name === 'string') {
        declared.set(sanitizeGeminiToolName(String(tool.name)), String(tool.name));
      }
    }

    const blocks: unknown[] = [];
    const text = this.fullText;
    if (text) blocks.push({ type: 'text', text });
    for (const call of this.toolCalls) {
      blocks.push({
        type: 'tool_use',
        id: `toolu_${randomUUID().replace(/-/g, '')}`,
        name: declared.get(call.name) || call.name,
        input: call.args,
        ...(call.signature ? { thought_signature: call.signature } : {}),
      });
    }
    if (blocks.length === 0) blocks.push({ type: 'text', text: '' });

    const stopReason: Anthropic.StopReason = this.toolCalls.length
      ? 'tool_use'
      : this.finishReason === 'MAX_TOKENS'
        ? 'max_tokens'
        : this.finishReason === 'SAFETY' || this.finishReason === 'RECITATION' || this.finishReason === 'BLOCKLIST'
          ? 'refusal'
          : 'end_turn';

    return makeMessage({
      model,
      content: blocks,
      stop_reason: stopReason,
      input_tokens: this.inputTokens,
      output_tokens: this.outputTokens,
    });
  }
}

// =============================================================================
// HTTP + SSE
// =============================================================================

async function readErrorBody(res: Response): Promise<string> {
  try {
    return (await res.text()).slice(0, 400);
  } catch {
    return '';
  }
}

async function postJSON(
  url: string,
  headers: Record<string, string>,
  body: unknown,
  signal?: AbortSignal,
): Promise<any> {
  const res = await fetch(url, {
    method: 'POST',
    headers,
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) {
    throw new ProviderRequestError(`LLM request failed (${res.status}): ${await readErrorBody(res)}`, res.status);
  }
  return res.json();
}

/** Yields the payload of each `data:` line in an SSE response. */
export async function* sseData(res: Response): AsyncGenerator<string, void, undefined> {
  const reader = res.body?.getReader();
  if (!reader) return;
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let newline = buffer.indexOf('\n');
      while (newline !== -1) {
        const line = buffer.slice(0, newline).trimEnd();
        buffer = buffer.slice(newline + 1);
        if (line.startsWith('data:')) {
          const payload = line.slice(5).trim();
          if (payload === '[DONE]') return;
          if (payload) yield payload;
        }
        newline = buffer.indexOf('\n');
      }
    }
  } finally {
    try {
      await reader.cancel();
    } catch {
      /* stream already closed */
    }
  }
}

async function openStream(
  url: string,
  headers: Record<string, string>,
  body: OpenAIChatBody | GeminiRequestBody,
  signal?: AbortSignal,
): Promise<Response> {
  const attempt = async (payload: unknown): Promise<Response> => {
    const res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
      signal,
    });
    return res;
  };

  let res = await attempt(body);
  if (!res.ok && res.status === 400 && body && (body as OpenAIChatBody).stream_options) {
    // Some OpenAI-compatible servers reject stream_options — retry without it once.
    const detail = await readErrorBody(res);
    if (/stream_options/i.test(detail)) {
      const retryBody = { ...(body as OpenAIChatBody) } as OpenAIChatBody;
      delete retryBody.stream_options;
      res = await attempt(retryBody);
    } else {
      throw new ProviderRequestError(`LLM stream failed (400): ${detail}`, 400);
    }
  }
  if (!res.ok) {
    throw new ProviderRequestError(`LLM stream failed (${res.status}): ${await readErrorBody(res)}`, res.status);
  }
  return res;
}

// =============================================================================
// STREAM BRIDGE
// =============================================================================

/**
 * Bridges an async backend stream onto the `on('text') / finalMessage()` surface the
 * agent uses. The backend starts on the first listener or on `finalMessage()`, so a
 * listener attached synchronously after `.stream()` never misses a delta.
 */
class StreamBridge implements LLMMessageStream {
  private listeners: Array<(delta: string, fullText: string) => void> = [];
  private started = false;
  private promise: Promise<AnthropicMessage> | null = null;

  constructor(private readonly run: (emit: (delta: string, fullText: string) => void) => Promise<AnthropicMessage>) {}

  on(event: 'text', listener: (delta: string, fullText: string) => void): this {
    if (event === 'text') this.listeners.push(listener);
    this.start();
    return this;
  }

  finalMessage(): Promise<AnthropicMessage> {
    this.start();
    return this.promise as Promise<AnthropicMessage>;
  }

  private start(): void {
    if (this.started) return;
    this.started = true;
    this.promise = this.run((delta, fullText) => {
      for (const listener of this.listeners) {
        try {
          listener(delta, fullText);
        } catch (err) {
          logger.debug({ err }, 'LLM stream text listener failed');
        }
      }
    });
    // Mark rejection as handled; consumers still observe it via finalMessage().
    this.promise.catch(() => undefined);
  }
}

// =============================================================================
// CLIENT FACTORY
// =============================================================================

function geminiUrl(baseUrl: string, model: string, stream: boolean): string {
  const prefixed = model.startsWith('models/') ? model : `models/${model}`;
  return `${baseUrl}/${prefixed}:${stream ? 'streamGenerateContent?alt=sse' : 'generateContent'}`;
}

/**
 * Build a client that speaks the Anthropic Messages surface while routing each request
 * to the provider resolved from its `model` field.
 */
export function createLLMClient(env: NodeJS.ProcessEnv = process.env): LLMClient {
  let anthropic: Anthropic | null = null;

  const getAnthropic = (): Anthropic => {
    const key = (env.ANTHROPIC_API_KEY || '').trim();
    if (!key) throw new MissingProviderKeyError(missingKeyMessage('anthropic', env));
    if (!anthropic) anthropic = new Anthropic({ apiKey: key });
    return anthropic;
  };

  const authHeaders = (provider: LLMProviderId): Record<string, string> => {
    const spec = PROVIDER_SPECS[provider];
    if (!spec.keyEnv) return { 'Content-Type': 'application/json' };
    const key = providerApiKey(spec, env);
    if (!key) throw new MissingProviderKeyError(missingKeyMessage(provider, env));
    return provider === 'gemini'
      ? { 'Content-Type': 'application/json', 'x-goog-api-key': key }
      : { 'Content-Type': 'application/json', Authorization: `Bearer ${key}` };
  };

  const dispatch = async (
    params: AnthropicParams,
    options?: LLMRequestOptions,
  ): Promise<AnthropicMessage> => {
    const route = resolveLLMRoute(params.model, env);

    if (route.provider === 'anthropic') {
      return getAnthropic().messages.create(
        { ...params, model: route.model },
        options as unknown as Anthropic.RequestOptions | undefined,
      );
    }

    if (route.provider === 'gemini') {
      const json = await postJSON(
        geminiUrl(providerBaseUrl('gemini', env), route.model, false),
        authHeaders('gemini'),
        buildGeminiRequestBody(params),
        options?.signal,
      );
      return messageFromGemini(json, route.model, params);
    }

    const json = await postJSON(
      `${providerBaseUrl(route.provider, env)}/chat/completions`,
      authHeaders(route.provider),
      buildOpenAIChatBody(params, route.model, false, route.provider),
      options?.signal,
    );
    return messageFromOpenAI(json, route.model);
  };

  const dispatchStream = (params: AnthropicParams): LLMMessageStream => {
    return new StreamBridge(async (emit) => {
      const route = resolveLLMRoute(params.model, env);

      if (route.provider === 'anthropic') {
        // Delegate to the SDK's own stream and re-emit its text events.
        const upstream = getAnthropic().messages.stream({ ...params, model: route.model });
        upstream.on('text', (delta: string, fullText: string) => emit(delta, fullText));
        return upstream.finalMessage();
      }

      if (route.provider === 'gemini') {
        const res = await openStream(
          geminiUrl(providerBaseUrl('gemini', env), route.model, true),
          authHeaders('gemini'),
          buildGeminiRequestBody(params),
        );
        const accumulator = new GeminiStreamAccumulator();
        for await (const payload of sseData(res)) {
          let chunk: any;
          try {
            chunk = JSON.parse(payload);
          } catch {
            continue;
          }
          const delta = accumulator.push(chunk);
          if (delta) emit(delta, accumulator.fullText);
        }
        return accumulator.finish(route.model, params);
      }

      const body = buildOpenAIChatBody(params, route.model, true, route.provider);
      const res = await openStream(
        `${providerBaseUrl(route.provider, env)}/chat/completions`,
        authHeaders(route.provider),
        body,
      );
      const accumulator = new OpenAIStreamAccumulator();
      for await (const payload of sseData(res)) {
        let chunk: any;
        try {
          chunk = JSON.parse(payload);
        } catch {
          continue;
        }
        const delta = accumulator.push(chunk);
        if (delta) emit(delta, accumulator.fullText);
      }
      return accumulator.finish(route.model);
    });
  };

  return {
    messages: {
      create: (params, options) => dispatch(params, options),
      stream: (params) => dispatchStream(params),
    },
  };
}
