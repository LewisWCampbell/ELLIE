/**
 * Offline verification of the Vertex provider wiring — no network, no GCP
 * credentials. Confirms the provider constructs and reports the right name, and
 * that the neutral→Anthropic message transform is shared correctly. It does NOT
 * make a real API call. Run: npx tsx src/llm/vertex.verify.ts
 */
import assert from "node:assert";
import { createVertexProvider } from "./vertex";
import { toAnthropicMessages, parseAnthropicResponse } from "./anthropicShared";

// AnthropicVertex resolves ADC lazily (only on the first request), so building
// the provider is safe without credentials.
process.env.VERTEX_PROJECT_ID ||= "verify-project";
const p = createVertexProvider();
assert.ok(p.name.startsWith("vertex:"), `provider name is vertex-scoped: ${p.name}`);
assert.ok(typeof p.chat === "function", "provider exposes chat()");

// Message transform round-trips user / assistant-with-tool-call / tool-result.
const msgs = toAnthropicMessages([
  { role: "user", text: "hi" },
  { role: "assistant", text: "", toolCalls: [{ id: "t1", name: "search_resume", input: { query: "skills" } }] },
  { role: "tool_results", results: [{ id: "t1", content: "found skills", isError: false }] },
]);
assert.strictEqual(msgs.length, 3, "three messages mapped");
assert.strictEqual(msgs[0].role, "user");
assert.ok(Array.isArray(msgs[1].content), "assistant content is blocks");
assert.ok(
  (msgs[1].content as Array<{ type: string }>).some((b) => b.type === "tool_use"),
  "assistant carries a tool_use block"
);

// Response parsing pulls text + tool calls out of an Anthropic-shaped message.
const parsed = parseAnthropicResponse({
  content: [
    { type: "text", text: "Sure!" },
    { type: "tool_use", id: "t2", name: "search_resume", input: { query: "projects" } },
  ],
  stop_reason: "tool_use",
} as never);
assert.strictEqual(parsed.text, "Sure!", "text extracted");
assert.strictEqual(parsed.toolCalls.length, 1, "one tool call extracted");
assert.strictEqual(parsed.toolCalls[0].name, "search_resume");
assert.strictEqual(parsed.refused, false, "not a refusal");

console.log("Vertex provider verification passed:");
console.log(`  ✓ provider constructs offline as ${p.name} (ADC resolved lazily, no key stored)`);
console.log("  ✓ shared message transform maps user / tool_use / tool_result");
console.log("  ✓ response parsing extracts text and tool calls");
