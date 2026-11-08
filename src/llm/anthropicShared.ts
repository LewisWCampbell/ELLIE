import Anthropic from "@anthropic-ai/sdk";
import { ChatResponse, NeutralMessage, ToolCall } from "./types";

/**
 * Message conversion + response parsing shared by both Anthropic-shaped
 * providers: the direct API (anthropic.ts, api.anthropic.com) and Vertex AI
 * (vertex.ts, Claude in Google Cloud). Both speak the same Messages API, so the
 * only real differences live in the client construction and request params.
 */

/**
 * Build the `system` blocks for a Messages request, skipping any empty text.
 * The Anthropic API rejects empty text content blocks ("system: text content
 * blocks must be non-empty"), and some channels (e.g. the realtime voice
 * cascade) pass an empty `systemDynamic`. The stable block, when present, keeps
 * the cache_control breakpoint so the persona prefix stays cacheable.
 */
export function buildSystemBlocks(stable: string, dynamic: string): Anthropic.TextBlockParam[] {
  const blocks: Anthropic.TextBlockParam[] = [];
  if (stable.trim()) {
    blocks.push({ type: "text", text: stable, cache_control: { type: "ephemeral" } });
  }
  if (dynamic.trim()) {
    blocks.push({ type: "text", text: dynamic });
  }
  return blocks;
}

/**
 * Convert provider-neutral history to Anthropic messages. With
 * `cacheLastMessage`, a cache_control breakpoint is attached to the final
 * content block of the final message, so the ENTIRE prefix (tools + system +
 * conversation so far) is cached and each following turn re-reads it at ~10%
 * of input price instead of full price — the conversation-caching pattern for
 * multi-turn calls. Only worth it when nothing volatile sits between the
 * cached system prefix and the messages (see anthropic.ts for the gate).
 */
export function toAnthropicMessages(
  messages: NeutralMessage[],
  cacheLastMessage = false
): Anthropic.MessageParam[] {
  const out = messages.map((msg): Anthropic.MessageParam => {
    switch (msg.role) {
      case "user":
        return { role: "user", content: msg.text };
      case "assistant": {
        // Prefer the raw content from the original response so thinking and
        // tool_use blocks are echoed back exactly as received.
        if (msg.raw) {
          return { role: "assistant", content: msg.raw as Anthropic.ContentBlockParam[] };
        }
        const blocks: Anthropic.ContentBlockParam[] = [];
        if (msg.text) blocks.push({ type: "text", text: msg.text });
        for (const call of msg.toolCalls) {
          blocks.push({ type: "tool_use", id: call.id, name: call.name, input: call.input });
        }
        return { role: "assistant", content: blocks };
      }
      case "tool_results":
        return {
          role: "user",
          content: msg.results.map(
            (r): Anthropic.ToolResultBlockParam => ({
              type: "tool_result",
              tool_use_id: r.id,
              content: r.content,
              is_error: r.isError,
            })
          ),
        };
    }
  });

  if (cacheLastMessage && out.length > 0) {
    const last = out[out.length - 1];
    // Only user-text and tool_result endings get the breakpoint (the request
    // always ends on one of those in our flows); cache_control is not valid on
    // e.g. thinking blocks, so anything else is left alone.
    if (typeof last.content === "string" && last.content.trim()) {
      last.content = [
        { type: "text", text: last.content, cache_control: { type: "ephemeral" } },
      ];
    } else if (Array.isArray(last.content) && last.content.length > 0) {
      const block = last.content[last.content.length - 1];
      if (block.type === "text" || block.type === "tool_result") {
        (block as Anthropic.TextBlockParam | Anthropic.ToolResultBlockParam).cache_control = {
          type: "ephemeral",
        };
      }
    }
  }
  return out;
}

/** Map an Anthropic Messages API response onto our provider-neutral shape. */
export function parseAnthropicResponse(response: Anthropic.Message): ChatResponse {
  const text = response.content
    .filter((b): b is Anthropic.TextBlock => b.type === "text")
    .map((b) => b.text)
    .join(" ")
    .trim();

  const toolCalls: ToolCall[] = response.content
    .filter((b): b is Anthropic.ToolUseBlock => b.type === "tool_use")
    .map((b) => ({
      id: b.id,
      name: b.name,
      input: b.input as Record<string, unknown>,
    }));

  return {
    text,
    toolCalls,
    refused: response.stop_reason === "refusal",
    raw: response.content,
  };
}
