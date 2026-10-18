/**
 * Summarization for context compaction — provider-agnostic (Anthropic by default,
 * any configured provider when the Anthropic key is absent).
 */

import { logger } from '../utils/logger';
import Anthropic from '@anthropic-ai/sdk';
import { createLLMClient, resolveLLMRoute, MissingProviderKeyError } from '../providers/llm-client';

export type SummarizerFn = (text: string, maxTokens: number) => Promise<string>;

interface ClaudeSummarizerOptions {
  apiKey?: string;
  model?: string;
}

const DEFAULT_SUMMARY_MODEL = process.env.AUTOPILOT_SUMMARY_MODEL || 'claude-haiku-4-5-20251001';

export function createClaudeSummarizer(options: ClaudeSummarizerOptions = {}): SummarizerFn | undefined {
  let model: string;
  try {
    // Resolves the route once: anthropic/claude-... natively, or the configured
    // provider's default model when only Gemini/OpenAI/etc. keys are present.
    model = resolveLLMRoute(options.model || DEFAULT_SUMMARY_MODEL).model;
  } catch (err) {
    if (err instanceof MissingProviderKeyError) {
      logger.debug({ err: err.message }, 'No summarizer provider available — compaction will truncate');
      return undefined;
    }
    throw err;
  }

  const client = createLLMClient();

  return async (text: string, maxTokens: number): Promise<string> => {
    const targetTokens = Math.max(128, Math.min(1200, Math.floor(maxTokens)));

    try {
      const response = await client.messages.create({
        model,
        max_tokens: targetTokens,
        system:
          'You are a summarizer that compresses conversation history for future context. '
          + 'Preserve key facts, decisions, constraints, and open questions. Be concise and structured.',
        messages: [
          {
            role: 'user',
            content:
              'Summarize the following conversation history for future context. '
              + 'Focus on durable facts and decisions.\n\n'
              + text,
          },
        ],
      });

      const summary = response.content
        .filter((b) => b.type === 'text')
        .map((b) => (b as Anthropic.TextBlock).text)
        .join('\n')
        .trim();

      return summary || '[Summary unavailable]';
    } catch (error) {
      logger.warn({ error }, 'Summarizer failed, falling back to naive summary');
      // Naive fallback: truncate the input.
      const maxChars = targetTokens * 4;
      return text.length > maxChars ? `${text.slice(0, maxChars)}\n\n[...truncated]` : text;
    }
  };
}
