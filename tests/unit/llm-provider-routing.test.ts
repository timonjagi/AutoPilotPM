import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  buildGeminiRequestBody,
  buildOpenAIChatBody,
  createLLMClient,
  defaultModelFor,
  describeLLMConfigProblem,
  inferProviderFromModel,
  isModelRefSupported,
  isProviderAvailable,
  listAvailableLLMProviders,
  messageFromGemini,
  messageFromOpenAI,
  MissingProviderKeyError,
  OpenAIStreamAccumulator,
  GeminiStreamAccumulator,
  resolveLLMRoute,
  sanitizeGeminiSchema,
  sanitizeGeminiToolName,
  toOpenAIMessages,
} from '../../src/providers/llm-client';
import { selectAdaptiveModel, stripProviderPrefix } from '../../src/models/adaptive';
import type Anthropic from '@anthropic-ai/sdk';

const anthropicParams = (): Anthropic.MessageCreateParamsNonStreaming =>
  ({
    model: 'claude-sonnet-4-6',
    max_tokens: 1024,
    system: [
      { type: 'text', text: 'core prompt', cache_control: { type: 'ephemeral' } },
      { type: 'text', text: 'dynamic context' },
    ],
    messages: [{ role: 'user', content: 'hello' }],
  }) as unknown as Anthropic.MessageCreateParamsNonStreaming;

// =============================================================================
// ROUTING
// =============================================================================

test('explicit provider/model prefix routes to that provider', () => {
  const env = { GEMINI_API_KEY: 'g-key' } as NodeJS.ProcessEnv;
  const route = resolveLLMRoute('gemini/gemini-2.5-flash', env);
  assert.deepEqual(route, { provider: 'gemini', model: 'gemini-2.5-flash', explicit: true });
});

test('explicit prefix for a provider without credentials throws', () => {
  const env = {} as NodeJS.ProcessEnv;
  assert.throws(() => resolveLLMRoute('gemini/gemini-2.5-flash', env), MissingProviderKeyError);
});

test('the default anthropic/ prefix falls back instead of blocking a Gemini-only setup', () => {
  // `anthropic/claude-opus-4-6` is the shipped DEFAULT_CONFIG primary, not something
  // the user typed — so adding only GEMINI_API_KEY must not fail the first message.
  const env = { GEMINI_API_KEY: 'g-key' } as NodeJS.ProcessEnv;
  const route = resolveLLMRoute('anthropic/claude-opus-4-6', env);
  assert.equal(route.provider, 'gemini');
  assert.equal(route.model, 'gemini-2.5-flash');
  assert.equal(route.explicit, false);

  // ...but an explicit prefix is still honoured when its provider IS configured.
  const both = { ANTHROPIC_API_KEY: 'a-key', GEMINI_API_KEY: 'g-key' } as NodeJS.ProcessEnv;
  const honoured = resolveLLMRoute('anthropic/claude-opus-4-6', both);
  assert.equal(honoured.provider, 'anthropic');
  assert.equal(honoured.explicit, true);
});

test('known model ids infer their provider', () => {
  assert.equal(inferProviderFromModel('claude-sonnet-4-6'), 'anthropic');
  assert.equal(inferProviderFromModel('gemini-2.5-flash'), 'gemini');
  assert.equal(inferProviderFromModel('gpt-4o'), 'openai');
  assert.equal(inferProviderFromModel('o3-mini'), 'openai');
  assert.equal(inferProviderFromModel('llama3'), null);
  assert.equal(inferProviderFromModel('meta-llama/Llama-3.3-70B-Instruct-Turbo'), null);
});

test('AUTOPILOT_LLM_PROVIDER forces the provider and swaps foreign model ids', () => {
  const env = {
    AUTOPILOT_LLM_PROVIDER: 'gemini',
    GEMINI_API_KEY: 'g-key',
    AUTOPILOT_GEMINI_MODEL: 'gemini-2.5-flash',
  } as NodeJS.ProcessEnv;

  const forced = resolveLLMRoute('claude-sonnet-4-6', env);
  assert.equal(forced.provider, 'gemini');
  assert.equal(forced.model, 'gemini-2.5-flash');

  const kept = resolveLLMRoute('gemini-2.0-flash', env);
  assert.equal(kept.provider, 'gemini');
  assert.equal(kept.model, 'gemini-2.0-flash');
});

test('a Claude-named model falls back to the configured provider when Anthropic is absent', () => {
  const env = { GEMINI_API_KEY: 'g-key' } as NodeJS.ProcessEnv;
  const route = resolveLLMRoute('claude-haiku-4-5-20251001', env);
  assert.equal(route.provider, 'gemini');
  assert.equal(route.model, 'gemini-2.5-flash');
  assert.equal(route.explicit, false);
});

test('Anthropic wins when both providers are configured', () => {
  const env = { ANTHROPIC_API_KEY: 'a-key', GEMINI_API_KEY: 'g-key' } as NodeJS.ProcessEnv;
  assert.deepEqual(listAvailableLLMProviders(env), ['anthropic', 'gemini']);
  assert.equal(resolveLLMRoute('claude-sonnet-4-6', env).provider, 'anthropic');
  assert.equal(resolveLLMRoute('gemini-2.5-flash', env).provider, 'gemini');
});

test('vendor-scoped model ids are not mistaken for a provider prefix', () => {
  const env = { TOGETHER_API_KEY: 't-key' } as NodeJS.ProcessEnv;
  const route = resolveLLMRoute('meta-llama/Llama-3.3-70B-Instruct-Turbo', env);
  assert.equal(route.provider, 'together');
  assert.equal(route.model, 'meta-llama/Llama-3.3-70B-Instruct-Turbo');
});

test('ollama is only available when OLLAMA_URL is set', () => {
  assert.equal(isProviderAvailable('ollama', {} as NodeJS.ProcessEnv), false);
  assert.equal(isProviderAvailable('ollama', { OLLAMA_URL: 'http://localhost:11434' } as NodeJS.ProcessEnv), true);
  const env = { OLLAMA_URL: 'http://localhost:11434' } as NodeJS.ProcessEnv;
  assert.equal(resolveLLMRoute('llama3', env).provider, 'ollama');
  assert.equal(defaultModelFor('ollama', env), 'llama3');
});

test('no provider at all reports a clear configuration problem', () => {
  const env = {} as NodeJS.ProcessEnv;
  const problem = describeLLMConfigProblem(env);
  assert.match(problem || '', /No LLM provider configured/);
  assert.throws(() => resolveLLMRoute('claude-sonnet-4-6', env), MissingProviderKeyError);
});

test('an unavailable forced provider is reported at startup', () => {
  const env = { AUTOPILOT_LLM_PROVIDER: 'gemini' } as NodeJS.ProcessEnv;
  assert.match(describeLLMConfigProblem(env) || '', /GEMINI_API_KEY/);
});

test('an unknown AUTOPILOT_LLM_PROVIDER value is reported', () => {
  const env = { AUTOPILOT_LLM_PROVIDER: 'mistral', ANTHROPIC_API_KEY: 'a-key' } as NodeJS.ProcessEnv;
  assert.match(describeLLMConfigProblem(env) || '', /not a known provider/);
});

test('valid configuration reports no problem', () => {
  const env = { ANTHROPIC_API_KEY: 'a-key' } as NodeJS.ProcessEnv;
  assert.equal(describeLLMConfigProblem(env), null);
});

test('model refs are accepted when a provider can serve them', () => {
  const env = { GEMINI_API_KEY: 'g-key' } as NodeJS.ProcessEnv;
  assert.equal(isModelRefSupported('gemini/gemini-2.5-flash', env), true);
  assert.equal(isModelRefSupported('claude-sonnet-4-6', env), true);
  assert.equal(isModelRefSupported('', env), false);
});

test('an unrecognised bare model id reroutes to the configured provider default', () => {
  // Same mechanism that lets Claude-named helper models (MEMORY_EXTRACT_MODEL)
  // run on a Gemini-only setup, so it must not be rejected.
  const env = { ANTHROPIC_API_KEY: 'a-key' } as NodeJS.ProcessEnv;
  assert.equal(isModelRefSupported('banana', env), true);
  const route = resolveLLMRoute('banana', env);
  assert.equal(route.provider, 'anthropic');
  assert.equal(route.model, 'claude-opus-4-6');
});

// =============================================================================
// OPENAI-COMPATIBLE TRANSLATION
// =============================================================================

test('anthropic messages translate to OpenAI roles, tool calls and tool results', () => {
  const params = {
    model: 'gpt-4o',
    max_tokens: 512,
    system: [{ type: 'text', text: 'be helpful' }],
    messages: [
      { role: 'user', content: 'find markets' },
      {
        role: 'assistant',
        content: [
          { type: 'text', text: 'searching' },
          { type: 'tool_use', id: 'toolu_1', name: 'search_markets', input: { query: 'btc' } },
        ],
      },
      {
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: 'toolu_1', content: '{"hits":1}' }],
      },
      { role: 'user', content: 'thanks' },
    ],
  } as unknown as Anthropic.MessageCreateParamsNonStreaming;

  const messages = toOpenAIMessages(params);
  assert.deepEqual(messages[0], { role: 'system', content: 'be helpful' });
  assert.deepEqual(messages[1], { role: 'user', content: 'find markets' });

  const assistant = messages[2] as { content: string | null; tool_calls: Array<{ id: string; function: { name: string; arguments: string } }> };
  assert.equal(assistant.content, 'searching');
  assert.equal(assistant.tool_calls.length, 1);
  assert.equal(assistant.tool_calls[0].id, 'toolu_1');
  assert.equal(assistant.tool_calls[0].function.name, 'search_markets');
  assert.deepEqual(JSON.parse(assistant.tool_calls[0].function.arguments), { query: 'btc' });

  assert.deepEqual(messages[3], { role: 'tool', tool_call_id: 'toolu_1', content: '{"hits":1}' });
  assert.deepEqual(messages[4], { role: 'user', content: 'thanks' });
});

test('openai chat body maps tools and only sends stream_options where supported', () => {
  const params = anthropicParams();
  params.tools = [
    {
      name: 'search_markets',
      description: 'Search prediction markets',
      input_schema: { type: 'object', properties: { query: { type: 'string' } }, required: ['query'] },
    } as unknown as Anthropic.Tool,
  ];

  const body = buildOpenAIChatBody(params, 'gpt-4o', true, 'openai');
  assert.deepEqual((body.tools as any[])[0].function.parameters, {
    type: 'object',
    properties: { query: { type: 'string' } },
    required: ['query'],
  });
  assert.deepEqual(body.stream_options, { include_usage: true });

  const ollamaBody = buildOpenAIChatBody(params, 'llama3', true, 'ollama');
  assert.equal(ollamaBody.stream_options, undefined);
});

test('openai responses convert to anthropic message shape', () => {
  const message = messageFromOpenAI(
    {
      id: 'chatcmpl-1',
      model: 'gpt-4o',
      choices: [
        {
          finish_reason: 'tool_calls',
          message: {
            role: 'assistant',
            content: null,
            tool_calls: [
              {
                id: 'call_1',
                type: 'function',
                function: { name: 'get_market', arguments: '{"market_id":"m1"}' },
              },
            ],
          },
        },
      ],
      usage: { prompt_tokens: 12, completion_tokens: 5 },
    },
    'gpt-4o',
  );

  assert.equal(message.stop_reason, 'tool_use');
  assert.equal(message.usage.input_tokens, 12);
  assert.equal(message.usage.output_tokens, 5);
  const block = message.content[0] as { type: string; name: string; input: Record<string, unknown> };
  assert.equal(block.type, 'tool_use');
  assert.equal(block.name, 'get_market');
  assert.deepEqual(block.input, { market_id: 'm1' });
});

test('openai finish_reason length maps to max_tokens', () => {
  const message = messageFromOpenAI(
    { choices: [{ finish_reason: 'length', message: { role: 'assistant', content: 'partial' } }] },
    'gpt-4o',
  );
  assert.equal(message.stop_reason, 'max_tokens');
  assert.equal((message.content[0] as { text: string }).text, 'partial');
});

test('openai stream accumulator folds text and tool call deltas', () => {
  const accumulator = new OpenAIStreamAccumulator();
  assert.equal(accumulator.push({ choices: [{ delta: { content: 'Hel' } }] }), 'Hel');
  assert.equal(accumulator.push({ choices: [{ delta: { content: 'lo' } }] }), 'lo');
  accumulator.push({
    choices: [
      {
        delta: {
          tool_calls: [
            { index: 0, id: 'call_9', function: { name: 'get_', arguments: '' } },
          ],
        },
      },
    ],
  });
  accumulator.push({
    choices: [
      {
        delta: {
          tool_calls: [
            { index: 0, function: { name: 'portfolio', arguments: '{"account' } },
          ],
        },
      },
    ],
  });
  accumulator.push({
    choices: [
      {
        delta: {
          tool_calls: [
            { index: 0, function: { arguments: '":"main"}' } },
          ],
        },
        finish_reason: 'tool_calls',
      },
    ],
  });

  const message = accumulator.finish('gpt-4o');
  assert.equal(accumulator.fullText, 'Hello');
  assert.equal(message.stop_reason, 'tool_use');
  assert.equal(message.content.length, 2);
  assert.equal((message.content[0] as { text: string }).text, 'Hello');
  const tool = message.content[1] as { type: string; id: string; name: string; input: Record<string, unknown> };
  assert.equal(tool.type, 'tool_use');
  assert.equal(tool.id, 'call_9');
  assert.equal(tool.name, 'get_portfolio');
  assert.deepEqual(tool.input, { account: 'main' });
});

// =============================================================================
// GEMINI TRANSLATION
// =============================================================================

test('anthropic messages translate to gemini contents, systemInstruction and functions', () => {
  const params = anthropicParams();
  params.tools = [
    {
      name: 'search_markets',
      description: 'Search prediction markets',
      input_schema: { type: 'object', properties: { query: { type: 'string' } } },
    } as unknown as Anthropic.Tool,
    {
      name: 'get_market',
      description: 'Get a market',
      input_schema: { type: 'object', properties: {}, additionalProperties: false },
    } as unknown as Anthropic.Tool,
  ] as Anthropic.Tool[];
  params.messages = [
    { role: 'user', content: 'find markets' },
    {
      role: 'assistant',
      content: [
        { type: 'text', text: 'searching' },
        { type: 'tool_use', id: 'toolu_1', name: 'search_markets', input: { query: 'btc' } },
      ],
    },
    {
      role: 'user',
      content: [{ type: 'tool_result', tool_use_id: 'toolu_1', content: '{"hits":1}' }],
    },
    { role: 'user', content: 'and gold?' },
  ] as Anthropic.MessageParam[];

  const body = buildGeminiRequestBody(params);

  assert.deepEqual(body.systemInstruction, {
    parts: [{ text: 'core prompt\n\ndynamic context' }],
  });
  assert.equal(body.contents[0].role, 'user');
  assert.deepEqual(body.contents[1].parts, [
    { text: 'searching' },
    { functionCall: { name: 'search_markets', args: { query: 'btc' } } },
  ]);
  // tool_result answers the function, then the extra user text merges into one user turn
  assert.deepEqual(body.contents[2].parts, [
    { functionResponse: { name: 'search_markets', response: { output: { hits: 1 } } } },
    { text: 'and gold?' },
  ]);
  assert.equal(body.contents.length, 3);

  const declarations = body.tools![0].functionDeclarations as Array<{ name: string; parameters?: Record<string, unknown> }>;
  assert.deepEqual(declarations[0].parameters, {
    type: 'object',
    properties: { query: { type: 'string' } },
  });
  assert.equal((declarations[1].parameters as any).additionalProperties, undefined);
});

test('gemini sanitizes tool names and schema keywords', () => {
  assert.equal(sanitizeGeminiToolName('my.tool'), 'my_tool');
  assert.equal(sanitizeGeminiToolName('1tool'), '_1tool');
  const schema = sanitizeGeminiSchema({
    type: 'object',
    $schema: 'http://json-schema.org/draft/2020-12/schema',
    title: 'x',
    properties: { a: { type: 'string', additionalProperties: false } },
  })!;
  assert.deepEqual(schema.properties.a, { type: 'string' });
  assert.equal(schema.$schema, undefined);
  assert.equal(schema.title, undefined);
});

test('gemini responses convert function calls to anthropic tool_use blocks', () => {
  const params = anthropicParams();
  params.tools = [
    { name: 'get_market', description: 'Get a market', input_schema: { type: 'object' } },
  ] as Anthropic.Tool[];

  const message = messageFromGemini(
    {
      candidates: [
        {
          content: {
            role: 'model',
            parts: [{ functionCall: { name: 'get_market', args: { market_id: 'm1' } } }],
          },
          finishReason: 'STOP',
        },
      ],
      usageMetadata: { promptTokenCount: 7, candidatesTokenCount: 3, totalTokenCount: 10 },
    },
    'gemini-2.5-flash',
    params,
  );

  assert.equal(message.stop_reason, 'tool_use');
  assert.equal(message.usage.input_tokens, 7);
  assert.equal(message.usage.output_tokens, 3);
  const block = message.content[0] as { type: string; name: string; input: Record<string, unknown> };
  assert.equal(block.name, 'get_market');
  assert.deepEqual(block.input, { market_id: 'm1' });
});

test('gemini stream accumulator folds text deltas', () => {
  const accumulator = new GeminiStreamAccumulator();
  assert.equal(accumulator.push({ candidates: [{ content: { parts: [{ text: 'Hi ' }] } }] }), 'Hi ');
  assert.equal(accumulator.push({ candidates: [{ content: { parts: [{ text: 'there' }] } }] }), 'there');
  const message = accumulator.finish('gemini-2.5-flash');
  assert.equal(message.stop_reason, 'end_turn');
  assert.equal((message.content[0] as { text: string }).text, 'Hi there');
});

// =============================================================================
// CLIENT DISPATCH (mocked network)
// =============================================================================

test('createLLMClient dispatches to the gemini endpoint', async () => {
  const calls: Array<{ url: string; headers: Record<string, string>; body: any }> = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input: any, init: any) => {
    calls.push({ url: String(input), headers: (init?.headers || {}) as Record<string, string>, body: JSON.parse(String(init?.body)) });
    return new Response(
      JSON.stringify({
        candidates: [
          {
            content: { role: 'model', parts: [{ text: 'hi from gemini' }] },
            finishReason: 'STOP',
          },
        ],
        usageMetadata: { promptTokenCount: 3, candidatesTokenCount: 4, totalTokenCount: 7 },
      }),
      { status: 200, headers: { 'content-type': 'application/json' } },
    );
  }) as typeof fetch;

  try {
    const env = { GEMINI_API_KEY: 'g-key' } as NodeJS.ProcessEnv;
    const client = createLLMClient(env);
    const response = await client.messages.create({
      model: 'gemini/gemini-2.5-flash',
      max_tokens: 100,
      messages: [{ role: 'user', content: 'hello' }],
    } as Anthropic.MessageCreateParamsNonStreaming);

    assert.equal(calls.length, 1);
    assert.equal(calls[0].url, 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent');
    assert.equal(calls[0].headers['x-goog-api-key'], 'g-key');
    assert.equal(calls[0].body.contents[0].parts[0].text, 'hello');
    assert.equal((response.content[0] as { text: string }).text, 'hi from gemini');
    assert.equal(response.stop_reason, 'end_turn');
    assert.equal(response.usage.input_tokens, 3);
    assert.equal(response.usage.output_tokens, 4);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('createLLMClient streams openai chunks and emits text events', async () => {
  const sse = [
    'data: {"choices":[{"delta":{"content":"stream"}}]}\n\n',
    'data: {"choices":[{"delta":{"content":"ed "}}]}\n\n',
    'data: {"choices":[{"delta":{"content":"answer"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\n',
    'data: [DONE]\n\n',
  ].join('');

  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async () =>
    new Response(sse, { status: 200, headers: { 'content-type': 'text/event-stream' } })) as typeof fetch;

  try {
    const env = { OPENAI_API_KEY: 'o-key' } as NodeJS.ProcessEnv;
    const client = createLLMClient(env);
    const stream = client.messages.stream({
      model: 'gpt-4o',
      max_tokens: 100,
      messages: [{ role: 'user', content: 'hello' }],
    } as Anthropic.MessageCreateParamsNonStreaming);

    const seen: string[] = [];
    stream.on('text', (_delta, fullText) => {
      seen.push(fullText);
    });

    const message = await stream.finalMessage();
    assert.deepEqual(seen, ['stream', 'streamed ', 'streamed answer']);
    assert.equal((message.content[0] as { text: string }).text, 'streamed answer');
    assert.equal(message.stop_reason, 'end_turn');
    assert.equal(message.usage.input_tokens, 5);
    assert.equal(message.usage.output_tokens, 2);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('provider http errors carry a status so retry can classify them', async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async () =>
    new Response('{"error":"rate limited"}', { status: 429, headers: { 'content-type': 'application/json' } })) as typeof fetch;

  try {
    const client = createLLMClient({ OPENAI_API_KEY: 'o-key' } as NodeJS.ProcessEnv);
    await assert.rejects(
      () =>
        client.messages.create({
          model: 'gpt-4o',
          max_tokens: 100,
          messages: [{ role: 'user', content: 'hello' }],
        } as Anthropic.MessageCreateParamsNonStreaming),
      (err: Error & { status?: number }) => {
        assert.equal(err.name, 'ProviderRequestError');
        assert.equal(err.status, 429);
        return true;
      },
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});

// =============================================================================
// MODEL SELECTION
// =============================================================================

test('stripProviderPrefix only removes a known provider prefix', () => {
  assert.equal(stripProviderPrefix('anthropic/claude-opus-4-6'), 'claude-opus-4-6');
  assert.equal(stripProviderPrefix('claude-opus-4-6'), 'claude-opus-4-6');
  assert.equal(stripProviderPrefix('meta-llama/Llama-3.3-70B-Instruct-Turbo'), 'meta-llama/Llama-3.3-70B-Instruct-Turbo');
});

test('adaptive selection keeps the provider prefix for routing', () => {
  const prefixed = selectAdaptiveModel({ primary: 'anthropic/claude-haiku-4-5-20251001', strategy: 'quality' });
  assert.equal(prefixed, 'anthropic/claude-haiku-4-5-20251001');

  const unknown = selectAdaptiveModel({ primary: 'gemini/gemini-2.5-flash', strategy: 'quality' });
  assert.equal(unknown, 'gemini/gemini-2.5-flash');
});
