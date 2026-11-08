import Anthropic from "@anthropic-ai/sdk";
import { AnthropicVertex } from "@anthropic-ai/vertex-sdk";
import { config } from "../config";
import { ChatRequest, ChatResponse, LlmProvider } from "./types";
import { buildSystemBlocks, parseAnthropicResponse, toAnthropicMessages } from "./anthropicShared";

/**
 * Claude on Google Vertex AI. Vertex speaks the same Messages API as the direct
 * Anthropic endpoint, but it lives inside Google Cloud, so calls are covered by
 * the same GCP BAA as your Cloud Run compute and Secret Manager — no separate
 * model-vendor BAA to sign. See docs/vertex-baa.md.
 *
 * Auth is Application Default Credentials (ADC): on Cloud Run/GCE the machine's
 * service account is used automatically; locally, run `gcloud auth
 * application-default login`. No API key is stored anywhere.
 */
export function createVertexProvider(): LlmProvider {
  // Built lazily on first use: constructing AnthropicVertex resolves Google
  // credentials (ADC) eagerly, so deferring it keeps process startup — and the
  // offline verify — from requiring credentials for a provider that may be idle.
  let client: AnthropicVertex | null = null;
  const getClient = () =>
    (client ??= new AnthropicVertex({
      projectId: config.vertex.projectId,
      region: config.vertex.region,
    }));

  return {
    name: `vertex:${config.vertex.model}`,
    async chat(req: ChatRequest): Promise<ChatResponse> {
      const response = await getClient().messages.create({
        model: config.vertex.model,
        max_tokens: req.maxTokens,
        // Note: no `output_config` here — that param is not yet available on
        // Vertex. Everything else matches the direct-API path.
        system: buildSystemBlocks(req.systemStable, req.systemDynamic),
        tools: req.tools.map(
          (t): Anthropic.Tool => ({
            name: t.name,
            description: t.description,
            input_schema: t.inputSchema as Anthropic.Tool.InputSchema,
          })
        ),
        messages: toAnthropicMessages(req.messages),
      });

      return parseAnthropicResponse(response);
    },
  };
}
