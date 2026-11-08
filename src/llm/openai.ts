import crypto from "crypto";
import { config } from "../config";
import {
  ChatRequest,
  ChatResponse,
  LlmProvider,
  NeutralMessage,
  ToolCall,
} from "./types";

/**
 * Adapter for any OpenAI-compatible chat-completions endpoint. This is the
 * cheap/open-model path: Hugging Face (router.huggingface.co/v1 or a dedicated
 * TGI Inference Endpoint), Together, Groq, Fireworks, vLLM, Ollama, or a
 * self-hosted fine-tune all speak this protocol.
 */

interface OpenAiToolCall {
  id?: string;
  type: string;
  function: { name: string; arguments: string };
}

interface OpenAiMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string | null;
  tool_calls?: OpenAiToolCall[];
  tool_call_id?: string;
}

function toOpenAiMessages(req: ChatRequest): OpenAiMessage[] {
  const out: OpenAiMessage[] = [
    { role: "system", content: `${req.systemStable}\n${req.systemDynamic}` },
  ];
  for (const msg of req.messages) {
    switch (msg.role) {
      case "user":
        out.push({ role: "user", content: msg.text });
        break;
      case "assistant": {
        if (msg.raw) {
          out.push(msg.raw as OpenAiMessage);
          break;
        }
        const m: OpenAiMessage = { role: "assistant", content: msg.text || null };
        if (msg.toolCalls.length > 0) {
          m.tool_calls = msg.toolCalls.map((c) => ({
            id: c.id,
            type: "function",
            function: { name: c.name, arguments: JSON.stringify(c.input) },
          }));
        }
        out.push(m);
        break;
      }
      case "tool_results":
        for (const r of msg.results) {
          out.push({
            role: "tool",
            tool_call_id: r.id,
            content: r.isError ? `ERROR: ${r.content}` : r.content,
          });
        }
        break;
    }
  }
  return out;
}

export function createOpenAiCompatibleProvider(): LlmProvider {
  const baseUrl = config.openaiBaseUrl.replace(/\/+$/, "");

  return {
    name: `openai-compatible:${config.openaiModel}`,
    async chat(req: ChatRequest): Promise<ChatResponse> {
      const res = await fetch(`${baseUrl}/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${config.openaiApiKey}`,
        },
        body: JSON.stringify({
          model: config.openaiModel,
          max_tokens: req.maxTokens,
          messages: toOpenAiMessages(req),
          tools: req.tools.map((t) => ({
            type: "function",
            function: {
              name: t.name,
              description: t.description,
              parameters: t.inputSchema,
            },
          })),
        }),
      });

      if (!res.ok) {
        const body = await res.text().catch(() => "");
        throw new Error(`LLM endpoint returned ${res.status}: ${body.slice(0, 500)}`);
      }

      const data = (await res.json()) as {
        choices?: Array<{
          message?: OpenAiMessage;
          finish_reason?: string;
        }>;
      };
      const message = data.choices?.[0]?.message;
      if (!message) {
        throw new Error("LLM endpoint returned no choices");
      }

      const toolCalls: ToolCall[] = (message.tool_calls ?? []).map((c) => {
        let input: Record<string, unknown> = {};
        try {
          input = JSON.parse(c.function.arguments || "{}");
        } catch {
          /* leave input empty; the tool handler will report the problem */
        }
        return {
          id: c.id ?? `call_${crypto.randomBytes(8).toString("hex")}`,
          name: c.function.name,
          input,
        };
      });

      return {
        text: (message.content ?? "").trim(),
        toolCalls,
        refused: data.choices?.[0]?.finish_reason === "content_filter",
        raw: message,
      };
    },
  };
}
