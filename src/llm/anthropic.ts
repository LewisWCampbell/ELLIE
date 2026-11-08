import Anthropic from "@anthropic-ai/sdk";
import { config } from "../config";
import { ChatRequest, ChatResponse, LlmProvider } from "./types";
import { buildSystemBlocks, parseAnthropicResponse, toAnthropicMessages } from "./anthropicShared";

/**
 * `output_config.effort` is only accepted on models that expose effort control
 * (the Opus 4.5+ and 4.6/4.7/4.8/5 families, Sonnet 5, Fable 5). Haiku 4.5 and
 * Sonnet 4.5 reject it with a 400 ("This model does not support the effort
 * parameter"), and the older Claude 1/2/3 lines never had it — so only send it
 * when the configured model supports it.
 */
function supportsEffort(model: string): boolean {
  if (/haiku/.test(model)) return false;
  if (/sonnet-4-5/.test(model)) return false;
  if (/claude-[123]-/.test(model)) return false;
  return true;
}

export function createAnthropicProvider(): LlmProvider {
  const client = new Anthropic({ apiKey: config.anthropicApiKey });

  function buildParams(req: ChatRequest): Anthropic.MessageCreateParamsNonStreaming {
    return {
      model: config.claudeModel,
      max_tokens: req.maxTokens,
      ...(supportsEffort(config.claudeModel)
        ? { output_config: { effort: "low" as const } }
        : {}),
      system: buildSystemBlocks(req.systemStable, req.systemDynamic),
      tools: req.tools.map(
        (t): Anthropic.Tool => ({
          name: t.name,
          description: t.description,
          input_schema: t.inputSchema as Anthropic.Tool.InputSchema,
        })
      ),
      // Conversation caching: also cache through the last message, so each
      // turn re-reads the growing history at ~10% of input price (and prefill
      // is faster). Callers opt in via cacheConversation when their dynamic
      // block is stable for the conversation (the voice cascade); otherwise
      // it's on only when systemDynamic is empty, because a per-turn dynamic
      // block (e.g. the SMS channel's current-time context) sits between the
      // cached system prefix and the messages and would invalidate this
      // breakpoint every turn, paying the 1.25x write with no reads.
      messages: toAnthropicMessages(
        req.messages,
        req.cacheConversation ?? !req.systemDynamic.trim()
      ),
    };
  }

  /** One [llm] line per call so cache effectiveness shows up in log pulls. */
  function logUsage(usage: Anthropic.Usage): void {
    const read = usage.cache_read_input_tokens ?? 0;
    const wrote = usage.cache_creation_input_tokens ?? 0;
    console.log(
      `[llm] tokens: in=${usage.input_tokens} cacheRead=${read} cacheWrite=${wrote} out=${usage.output_tokens}`
    );
  }

  return {
    name: `anthropic:${config.claudeModel}`,
    async chat(req: ChatRequest): Promise<ChatResponse> {
      const response = await client.messages.create(buildParams(req));
      logUsage(response.usage);
      return parseAnthropicResponse(response);
    },
    async chatStream(req: ChatRequest, onText: (delta: string) => void): Promise<ChatResponse> {
      const stream = client.messages.stream(buildParams(req));
      stream.on("text", (delta) => onText(delta));
      const response = await stream.finalMessage();
      logUsage(response.usage);
      return parseAnthropicResponse(response);
    },
  };
}
