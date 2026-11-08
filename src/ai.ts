/**
 * The turn engine for the TEXT-mediated channels (cascaded voice and website
 * chat): one caller message in, one grounded reply out. Wraps the shared LLM
 * provider with the persona, the per-caller context, and the tool loop
 * (search_resume RAG, save_user_fact, book_intro_call, and the web-pairing
 * show_work / show_choices tools). The realtime phone path has its own turn
 * loop inside deepgramCascade.ts; this one serves everything turn-based.
 */
import { config } from "./config";
import * as db from "./db";
import { CallSession, classifyShowUrl, isPaired, pushWebEvent, pushFactsSnapshot } from "./sessions";
import { provider } from "./llm/provider";
import { ToolCall, ToolDef, ToolResultMsg } from "./llm/types";
import { personaCore, personaContext, SEARCH_RESUME } from "./persona";
import { searchKnowledge, formatRetrieved } from "./knowledge";
import { BOOK_INTRO_CALL, handleBookIntroCall } from "./booking/booking";
import { CHECK_AVAILABILITY, handleCheckAvailability } from "./booking/availability";

console.log(`LLM provider: ${provider.name} | mode: ${config.botMode}`);

const TOOLS: ToolDef[] = [
  { name: SEARCH_RESUME.name, description: SEARCH_RESUME.description, inputSchema: SEARCH_RESUME.parameters },
  {
    name: "save_user_fact",
    description:
      "Save a piece of information about the caller so it is remembered across calls — " +
      "their name, who they work for, why they're calling, preferences. Save one concise fact.",
    inputSchema: {
      type: "object",
      properties: {
        category: { type: "string", description: "Short label, e.g. name, company, role, reason_for_calling." },
        fact: { type: "string", description: "A short standalone sentence. E.g. 'Their name is Priya.'" },
      },
      required: ["category", "fact"],
    },
  },
  {
    name: "show_work",
    description:
      "Display a piece of Lewis's work in the caller's paired web browser (only works after they pair). " +
      "Prefer the LIVE, interactive URL for a project when the visuals list has one — the caller can " +
      "actually use it while you talk — and fall back to a screenshot when it doesn't. " +
      "After showing something, ask a question about it. HTTPS URL from the visuals list only.",
    inputSchema: {
      type: "object",
      properties: {
        url: { type: "string", description: "Publicly accessible HTTPS URL: a live page or an image." },
        caption: { type: "string", description: "Short caption naming what they're looking at." },
      },
      required: ["url", "caption"],
    },
  },
  {
    name: "show_choices",
    description:
      "Display a question with clickable options in the caller's paired browser (only after they pair). " +
      "The caller's click is reported on their next turn.",
    inputSchema: {
      type: "object",
      properties: {
        question: { type: "string", description: "The question to display." },
        options: { type: "array", items: { type: "string" }, description: "2–6 short options." },
      },
      required: ["question", "options"],
    },
  },
  { name: CHECK_AVAILABILITY.name, description: CHECK_AVAILABILITY.description, inputSchema: CHECK_AVAILABILITY.parameters },
  { name: BOOK_INTRO_CALL.name, description: BOOK_INTRO_CALL.description, inputSchema: BOOK_INTRO_CALL.parameters },
  {
    name: "take_message",
    description:
      "Record a message for the owner — use when the caller asks something you can't answer from " +
      "the resume, wants a callback, or wants to leave a note. Tell the caller you'll pass it along.",
    inputSchema: {
      type: "object",
      properties: {
        message: { type: "string", description: "The message, including who it's from and how to reach them." },
      },
      required: ["message"],
    },
  },
  {
    name: "end_call",
    description:
      "Hang up the phone call when the caller says goodbye. Your text response is spoken as the farewell.",
    inputSchema: { type: "object", properties: {} },
  },
];

async function handleToolCall(session: CallSession, call: ToolCall): Promise<ToolResultMsg> {
  const input = call.input;
  let content: string;
  let isError = false;

  switch (call.name) {
    case "check_availability": {
      content = await handleCheckAvailability(input);
      isError = content.startsWith("Error");
      break;
    }
    case "book_intro_call": {
      content = await handleBookIntroCall(input, {
        fromNumber: session.callerNumber,
        callSid: session.callSid,
        notify: (s) => pushWebEvent(session, { type: "fact", category: "booking", fact: s }),
      });
      isError = content.startsWith("Error");
      break;
    }
    case "search_resume": {
      const results = searchKnowledge(String(input.query ?? ""), 4);
      // Surface the sources on the paired browser so the screen tracks the talk.
      for (const r of results) {
        pushWebEvent(session, { type: "fact", category: "resume", fact: `${r.heading}` });
      }
      content = formatRetrieved(results);
      break;
    }
    case "save_user_fact": {
      const category = String(input.category ?? "general");
      const fact = String(input.fact ?? "");
      if (!fact) {
        content = "Error: fact was empty.";
        isError = true;
        break;
      }
      db.saveFact(session.callerNumber, category, fact, session.callSid);
      pushWebEvent(session, { type: "fact", category, fact });
      pushFactsSnapshot(session);
      content = "Saved.";
      break;
    }
    case "take_message": {
      const message = String(input.message ?? "");
      if (!message) {
        content = "Error: message was empty.";
        isError = true;
        break;
      }
      db.saveFact(session.callerNumber, "message", message, session.callSid);
      pushWebEvent(session, { type: "fact", category: "message", fact: message });
      content = "Message recorded. Let the caller know it will be passed along.";
      break;
    }
    case "show_work": {
      if (!isPaired(session)) {
        content = "Error: the caller hasn't paired a browser yet. Invite them to the website with the pairing code.";
        isError = true;
        break;
      }
      const url = String(input.url ?? input.image_url ?? "");
      if (!/^https:\/\//i.test(url)) {
        content = "Error: url must be a public HTTPS URL.";
        isError = true;
        break;
      }
      const kind = classifyShowUrl(url, config.siteUrl);
      pushWebEvent(session, { type: "show", kind, url, caption: String(input.caption ?? "") });
      content =
        kind === "page"
          ? "The live page is now running in the caller's browser; they can interact with it while you talk."
          : kind === "link"
            ? "That site refuses embedding, so the caller sees a card with a button to open it in a new tab."
            : "Image is now displayed in the caller's browser.";
      break;
    }
    case "show_choices": {
      if (!isPaired(session)) {
        content = "Error: the caller hasn't paired a browser yet. Invite them to the website with the pairing code.";
        isError = true;
        break;
      }
      const options = Array.isArray(input.options) ? input.options.map(String) : [];
      if (options.length < 2) {
        content = "Error: provide at least two options.";
        isError = true;
        break;
      }
      pushWebEvent(session, { type: "choices", question: String(input.question ?? ""), options });
      content = "Choices are now displayed in the caller's browser.";
      break;
    }
    case "end_call": {
      session.shouldHangup = true;
      content = "The call will end after your farewell is spoken.";
      break;
    }
    default:
      content = `Error: unknown tool ${call.name}.`;
      isError = true;
  }

  return { id: call.id, content, isError };
}

/**
 * Run one conversational turn: feed the caller's words to the model, execute
 * any tool calls, and return the text to speak back.
 */
export async function runTurn(session: CallSession, callerText: string): Promise<string> {
  const webNotes = session.pendingWebNotes.splice(0);
  const noteBlock =
    webNotes.length > 0 ? `\n\n<web_session_events>\n${webNotes.join("\n")}\n</web_session_events>` : "";

  session.messages.push({ role: "user", text: callerText + noteBlock });

  let spoken = "";
  for (let iteration = 0; iteration < 6; iteration++) {
    const response = await provider.chat({
      systemStable: personaCore("voice"),
      systemDynamic: personaContext(session.knownFacts, session.pairingCode),
      messages: session.messages,
      tools: TOOLS,
      maxTokens: 2048,
    });

    session.messages.push({
      role: "assistant",
      text: response.text,
      toolCalls: response.toolCalls,
      raw: response.raw,
    });

    if (response.refused) {
      return "I'm sorry, I can't help with that. Is there something else you'd like to know?";
    }

    if (response.text) spoken = response.text;

    if (response.toolCalls.length === 0) break;

    const results = await Promise.all(response.toolCalls.map((call) => handleToolCall(session, call)));
    session.messages.push({ role: "tool_results", results });

    if (session.shouldHangup && spoken) break;
  }

  return spoken || "Sorry, I lost my train of thought. Could you say that again?";
}
