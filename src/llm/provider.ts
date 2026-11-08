import { config } from "../config";
import { createAnthropicProvider } from "./anthropic";
import { createVertexProvider } from "./vertex";
import { createOpenAiCompatibleProvider } from "./openai";
import { LlmProvider } from "./types";

/**
 * The configured text LLM, shared by the cascaded voice loop (ai.ts) and the
 * website chat (webchat.ts):
 *  - "anthropic" — Claude via api.anthropic.com (default)
 *  - "vertex"    — Claude via Google Vertex AI (covered by your GCP BAA)
 *  - "openai"    — any OpenAI-compatible endpoint (Hugging Face, Groq, Ollama, vLLM)
 */
function selectProvider(): LlmProvider {
  switch (config.llmProvider) {
    case "vertex":
      return createVertexProvider();
    case "openai":
      return createOpenAiCompatibleProvider();
    default:
      return createAnthropicProvider();
  }
}

export const provider: LlmProvider = selectProvider();
