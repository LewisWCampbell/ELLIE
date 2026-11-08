/**
 * Provider-neutral chat types so the conversation loop works with either the
 * Anthropic API or any OpenAI-compatible endpoint (Hugging Face TGI/router,
 * Together, Groq, Fireworks, a self-hosted fine-tune, ...).
 */

export interface ToolDef {
  name: string;
  description: string;
  /** JSON Schema for the tool input. */
  inputSchema: Record<string, unknown>;
}

export interface ToolCall {
  id: string;
  name: string;
  input: Record<string, unknown>;
}

export interface ToolResultMsg {
  id: string;
  content: string;
  isError: boolean;
}

export type NeutralMessage =
  | { role: "user"; text: string }
  | {
      role: "assistant";
      text: string;
      toolCalls: ToolCall[];
      /**
       * Provider-native assistant content, kept so the same provider can
       * replay the turn byte-exact (Anthropic requires thinking/tool_use
       * blocks to be echoed back unchanged).
       */
      raw?: unknown;
    }
  | { role: "tool_results"; results: ToolResultMsg[] };

export interface ChatRequest {
  /** Stable part of the system prompt (cacheable). */
  systemStable: string;
  /** Per-call part of the system prompt (facts, pairing code). */
  systemDynamic: string;
  messages: NeutralMessage[];
  tools: ToolDef[];
  maxTokens: number;
  /**
   * Cache the conversation too (a breakpoint on the last message), so each
   * turn re-reads the whole prefix at cache-read price. Only set this when
   * systemDynamic is byte-stable across the conversation's turns — a dynamic
   * block that changes per turn (e.g. a live timestamp) sits before the
   * messages and would invalidate the breakpoint every time, paying cache
   * writes with no reads. Defaults to on only when systemDynamic is empty.
   */
  cacheConversation?: boolean;
}

export interface ChatResponse {
  text: string;
  toolCalls: ToolCall[];
  /** True when the provider refused to answer (safety). */
  refused: boolean;
  /** Provider-native assistant content to store on the history entry. */
  raw: unknown;
}

export interface LlmProvider {
  name: string;
  chat(req: ChatRequest): Promise<ChatResponse>;
  /**
   * Like chat(), but invokes `onText` with each text delta as it is generated.
   * Optional: latency-sensitive callers (the voice cascade) use it to start
   * speaking before the full reply exists; providers without it fall back to
   * chat().
   */
  chatStream?(req: ChatRequest, onText: (delta: string) => void): Promise<ChatResponse>;
}
