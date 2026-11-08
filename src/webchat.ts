import { provider } from "./llm/provider";
import { NeutralMessage, ToolDef, ToolResultMsg } from "./llm/types";
import { personaCore, personaContext, SEARCH_RESUME } from "./persona";
import { searchKnowledge, formatRetrieved } from "./knowledge";
import { BOOK_INTRO_CALL, handleBookIntroCall } from "./booking/booking";
import { CHECK_AVAILABILITY, handleCheckAvailability } from "./booking/availability";
import { logQuestion } from "./gaps";

/**
 * The website chat brain: same persona/template and same RAG as the phone bot,
 * but text in / text out for an anonymous web visitor. Powers /api/chat.
 */

export interface ChatTurn {
  role: "user" | "assistant";
  text: string;
}

export interface ChatResult {
  reply: string;
  sources: Array<{ source: string; heading: string }>;
}

const TOOLS: ToolDef[] = [
  { name: SEARCH_RESUME.name, description: SEARCH_RESUME.description, inputSchema: SEARCH_RESUME.parameters },
  { name: CHECK_AVAILABILITY.name, description: CHECK_AVAILABILITY.description, inputSchema: CHECK_AVAILABILITY.parameters },
  { name: BOOK_INTRO_CALL.name, description: BOOK_INTRO_CALL.description, inputSchema: BOOK_INTRO_CALL.parameters },
];

function toNeutral(history: ChatTurn[]): NeutralMessage[] {
  return history.slice(-10).map((t): NeutralMessage =>
    t.role === "user"
      ? { role: "user", text: t.text }
      : { role: "assistant", text: t.text, toolCalls: [] }
  );
}

export async function answer(message: string, history: ChatTurn[] = []): Promise<ChatResult> {
  logQuestion("web", message);
  const messages: NeutralMessage[] = [...toNeutral(history), { role: "user", text: message }];
  const sources: Array<{ source: string; heading: string }> = [];

  let reply = "";
  for (let iteration = 0; iteration < 4; iteration++) {
    const response = await provider.chat({
      systemStable: personaCore("web"),
      systemDynamic: personaContext([]),
      messages,
      tools: TOOLS,
      maxTokens: 1024,
    });

    messages.push({ role: "assistant", text: response.text, toolCalls: response.toolCalls, raw: response.raw });

    if (response.refused) return { reply: "Sorry, I can't help with that one.", sources };
    if (response.text) reply = response.text;
    if (response.toolCalls.length === 0) break;

    const results: ToolResultMsg[] = await Promise.all(
      response.toolCalls.map(async (call): Promise<ToolResultMsg> => {
        if (call.name === "search_resume") {
          const found = searchKnowledge(String(call.input.query ?? ""), 4);
          for (const r of found) sources.push({ source: r.source, heading: r.heading });
          return { id: call.id, content: formatRetrieved(found), isError: false };
        }
        if (call.name === "check_availability") {
          const content = await handleCheckAvailability(call.input);
          return { id: call.id, content, isError: content.startsWith("Error") };
        }
        if (call.name === "book_intro_call") {
          // Anonymous web visitor: no phone number, no paired browser.
          const content = await handleBookIntroCall(call.input, {});
          return { id: call.id, content, isError: content.startsWith("Error") };
        }
        return { id: call.id, content: `Unknown tool: ${call.name}`, isError: true };
      })
    );
    messages.push({ role: "tool_results", results });
  }

  // De-duplicate sources by source+heading.
  const seen = new Set<string>();
  const uniqueSources = sources.filter((s) => {
    const key = `${s.source}#${s.heading}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });

  return { reply: reply || "Sorry, I didn't catch that — could you rephrase?", sources: uniqueSources };
}
