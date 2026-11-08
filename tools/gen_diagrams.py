#!/usr/bin/env python3
"""Generate theme-aware inline-SVG sequence diagrams + an ERD for the Elle docs.

Output: HTML fragments in svg/, each a <figure class="figwrap"> wrapping an
<svg class="sdg"> that styles itself via the docs site's CSS variables, so
every diagram follows light/dark automatically.
"""
import html, os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "svg")
CH = 6.0  # approx px per char at 11px monospace-ish
# Note text is the same 11px monospace but italic, which runs wider. Measured
# against the rendered output rather than guessed: at 6.0 the box came out
# narrower than its own text and the words ran past the border.
NOTE_CH = 6.7
NOTE_PAD = 30  # 15px breathing room each side of the text


def esc(s):
    return html.escape(s, quote=True)


def seq(name, title, actors, steps, note_w=None):
    """actors: list of (id, label). steps: list of tuples:
    ("msg", frm, to, label)        solid arrow
    ("ret", frm, to, label)        dashed return
    ("self", actor, label)         self-loop
    ("note", actor_or_(a,b), text) note box centered on lane(s)
    """
    n = len(actors)
    lane_w = max(128, min(160, 840 // n))
    lanes_w = lane_w * n + 20

    # Widen the canvas if a note needs more room than the lanes provide, and
    # centre the lanes in what is left. Clamping the note box to the lane width
    # instead (which is what this used to do) shrinks the BOX but not the TEXT,
    # so the words run out past the border and, on the widest notes, past the
    # edge of the SVG itself. Growing the canvas is the only fix that keeps the
    # text inside its own box.
    def note_width(text):
        return int(len(text) * NOTE_CH) + NOTE_PAD

    widest_note = max(
        (note_width(st[2]) for st in steps if st[0] == "note"),
        default=0,
    )
    W = max(lanes_w, (note_w or widest_note) + 24)
    off = (W - lanes_w) // 2

    x = {a[0]: 10 + lane_w * i + lane_w // 2 + off for i, a in enumerate(actors)}
    y = 74
    rows = []
    for st in steps:
        if st[0] == "note":
            rows.append((y, st)); y += 40
        elif st[0] == "self":
            rows.append((y, st)); y += 44
        else:
            rows.append((y, st)); y += 34
    H = y + 16
    p = []
    p.append(f'<svg class="sdg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="{esc(title)}">')
    p.append(f'<defs><marker id="ah-{name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" class="sdg-ah"/></marker></defs>')
    # lifelines first (under everything)
    for aid, _ in actors:
        p.append(f'<line class="sdg-life" x1="{x[aid]}" y1="46" x2="{x[aid]}" y2="{H-8}"/>')
    # actor boxes
    for i, (aid, label) in enumerate(actors):
        bx = 10 + lane_w * i + 6 + off
        bw = lane_w - 12
        p.append(f'<g class="sdg-actor"><rect x="{bx}" y="8" width="{bw}" height="30" rx="7"/>'
                 f'<text x="{x[aid]}" y="27" text-anchor="middle">{esc(label)}</text></g>')
    for ry, st in rows:
        kind = st[0]
        if kind in ("msg", "ret"):
            _, frm, to, label = st
            x1, x2 = x[frm], x[to]
            sgn = 1 if x2 > x1 else -1
            cls = "sdg-msg" if kind == "msg" else "sdg-ret"
            p.append(f'<line class="{cls}" x1="{x1}" y1="{ry}" x2="{x2 - sgn*4}" y2="{ry}" marker-end="url(#ah-{name})"/>')
            p.append(f'<text class="sdg-lbl" x="{(x1+x2)//2}" y="{ry-6}" text-anchor="middle">{esc(label)}</text>')
        elif kind == "self":
            _, actor, label = st
            ax = x[actor]
            p.append(f'<path class="sdg-msg" d="M {ax} {ry} h 34 v 18 h -30" fill="none" marker-end="url(#ah-{name})"/>')
            p.append(f'<text class="sdg-lbl" x="{ax+42}" y="{ry+12}" text-anchor="start">{esc(label)}</text>')
        elif kind == "note":
            _, target, text = st
            if isinstance(target, tuple):
                cx = (x[target[0]] + x[target[1]]) // 2
            else:
                cx = x[target]
            # W was sized above to fit the widest note, so this never has to
            # clamp the box below what its text needs.
            w = note_w or max(180, note_width(text))
            nx = max(12, min(cx - w // 2, W - w - 12))
            p.append(f'<g class="sdg-note"><rect x="{nx}" y="{ry-2}" width="{w}" height="26" rx="5"/>'
                     f'<text x="{nx + w//2}" y="{ry+15}" text-anchor="middle">{esc(text)}</text></g>')
    p.append('</svg>')
    return "".join(p)


def entity(x, y, name, cols, w=196):
    rh = 19
    h = 26 + rh * len(cols)
    p = [f'<g class="erd-ent"><rect class="erd-box" x="{x}" y="{y}" width="{w}" height="{h}" rx="8"/>']
    p.append(f'<rect class="erd-head" x="{x}" y="{y}" width="{w}" height="24" rx="8"/>')
    p.append(f'<rect class="erd-head" x="{x}" y="{y+14}" width="{w}" height="10"/>')
    p.append(f'<text class="erd-title" x="{x+w/2}" y="{y+17}" text-anchor="middle">{esc(name)}</text>')
    for i, (cn, ct, marker) in enumerate(cols):
        cy = y + 26 + rh * i + 13
        mark = {"PK": "PK ", "FK": "→ ", "": "   "}[marker]
        cls = "erd-col erd-pk" if marker == "PK" else ("erd-col erd-fk" if marker == "FK" else "erd-col")
        p.append(f'<text class="{cls}" x="{x+10}" y="{cy}">{esc(mark + cn)}</text>')
        p.append(f'<text class="erd-type" x="{x+w-10}" y="{cy}" text-anchor="end">{esc(ct)}</text>')
    p.append('</g>')
    return "".join(p), h


def figure(svg, caption):
    return f'<figure class="figwrap">{svg}<figcaption>{caption}</figcaption></figure>\n'


def write(name, content):
    with open(os.path.join(OUT, name), "w") as f:
        f.write(content)
    print("wrote", name)


# ---------------------------------------------------------------- sequences
write("seq_setup.html", figure(seq("setup", "Call setup and the uninterruptible greeting",
    [("c", "Caller"), ("tw", "Twilio"), ("rt", "/voice-realtime"), ("br", "Bridge"), ("cs", "Cascade")],
    [
        ("msg", "c", "tw", "dials the number"),
        ("msg", "tw", "rt", "POST webhook (signed)"),
        ("note", "rt", "verify Twilio signature · assess caller tier (normal / limited / blocked)"),
        ("ret", "rt", "tw", "TwiML <Connect><Stream> + HMAC media token"),
        ("msg", "tw", "br", "WS /media · start frame"),
        ("note", "br", "verifyMediaToken: recompute HMAC, 5-min age, constant-time compare"),
        ("msg", "br", "cs", "create model (static persona + per-call context)"),
        ("self", "cs", "greet(): deterministic line, no LLM round-trip"),
        ("ret", "cs", "br", "greeting audio, streamed as it synthesizes"),
        ("msg", "br", "c", "paced 20 ms frames · first audio ≈300 ms"),
        ("note", ("br", "cs"), "barge-in stays locked until every greeting frame has been SENT"),
    ]),
    "Call setup: signature gate, media-token gate, then a greeting that always plays to the end."))

write("seq_turn.html", figure(seq("turn", "One spoken turn through the cascade",
    [("c", "Caller"), ("st", "Deepgram STT"), ("cs", "Cascade"), ("ll", "Claude"), ("tt", "Aura TTS"), ("br", "Bridge")],
    [
        ("msg", "c", "st", "μ-law audio, streamed"),
        ("ret", "st", "cs", "interim transcripts (echo-filtered)"),
        ("msg", "st", "cs", "speech_final → utterance + language tag"),
        ("self", "cs", "merge queued utterances into ONE turn"),
        ("msg", "cs", "tt", "warm(): connect overlaps thinking"),
        ("msg", "cs", "ll", "chatStream (cached prefix + conversation)"),
        ("ret", "ll", "cs", "tokens… first full sentence"),
        ("msg", "cs", "tt", "Speak(sentence 1) while generation continues"),
        ("ret", "tt", "cs", "audio chunks → exact 160-byte frames"),
        ("msg", "cs", "br", "onAudioDelta per 20 ms frame"),
        ("msg", "br", "c", "drift-corrected playback · ≈1 s to first word"),
    ]),
    "Sentence streaming is the latency lever: Elle starts speaking while the rest of the reply is still being generated."))

write("seq_barge.html", figure(seq("barge", "Barge-in: three cancellations from one integer",
    [("c", "Caller"), ("st", "Deepgram STT"), ("cs", "Cascade"), ("br", "Bridge"), ("tw", "Twilio")],
    [
        ("msg", "c", "st", "talks over Elle"),
        ("note", "st", "raw VAD ignored: needs ≥6 chars of recognized, non-echo speech"),
        ("msg", "st", "cs", "barge-in trigger"),
        ("msg", "cs", "br", "onSpeechStarted"),
        ("note", "br", "greeting gate: ignored until the opening line has fully played"),
        ("msg", "br", "cs", "cancelResponse() → turnSeq++"),
        ("self", "cs", "LLM loop + TTS pump bail at next checkpoint"),
        ("msg", "br", "tw", "clear → flush queued playback"),
        ("msg", "br", "cs", "truncate(itemId, playedMs)"),
        ("note", ("cs", "br"), "context now matches exactly what the caller actually heard"),
    ]),
    "The interrupted turn doesn't error; it quietly stops being real. The caller's words become the next turn."))

write("seq_booking.html", figure(seq("booking", "Tentative booking and the consent flow",
    [("c", "Caller"), ("el", "Elle (model)"), ("bk", "booking.ts"), ("gc", "Google Calendar"), ("sm", "sms.ts")],
    [
        ("msg", "c", "el", '"Tuesday at 2 works" + name + email'),
        ("ret", "el", "c", "reads date, time, timezone back → clear yes"),
        ("msg", "el", "bk", "book_intro_call(name, email, start ISO+offset)"),
        ("self", "bk", "validate email shape + timestamp"),
        ("msg", "bk", "gc", "freeBusy(start, +30 min)"),
        ("ret", "gc", "bk", "slot free"),
        ("msg", "bk", "gc", 'insert event "pending confirmation"'),
        ("self", "bk", "bookings row · status = pending"),
        ("ret", "bk", "el", "tool result: TENTATIVE, say owner confirms by email"),
        ("msg", "el", "c", "full consent script → plain yes-or-no"),
        ("msg", "el", "sm", 'save_user_fact("sms_consent") → confirmation SMS'),
        ("note", ("bk", "sm"), "consent fact + confirmation text = the audit trail; STOP/HELP enforced by Twilio upstream"),
    ]),
    "The AI never finalizes: every hold is pending until the owner confirms, and texts require the spoken consent script."))

write("seq_pair.html", figure(seq("pair", "Pairing a browser with a live call",
    [("c", "Caller (phone)"), ("cs", "Cascade"), ("se", "CallSession"), ("bw", "Browser")],
    [
        ("msg", "c", "cs", '"can I see that on screen?"'),
        ("ret", "cs", "c", "speaks the pairing code, digit by digit"),
        ("msg", "bw", "se", "POST /api/pair {code} · 10/min/IP"),
        ("ret", "se", "bw", "WS /ws joined to this call's session"),
        ("msg", "cs", "se", "both sides' transcripts as they happen"),
        ("ret", "se", "bw", "live transcript events"),
        ("msg", "cs", "se", "show_image / show_choices tool output"),
        ("msg", "bw", "se", "typed message (length-capped)"),
        ("msg", "se", "cs", "injectText → queued as a real turn"),
        ("ret", "cs", "c", "Elle answers OUT LOUD on the call"),
        ("note", ("se", "bw"), "session dies with the call; codes are unusable after hangup"),
    ]),
    "Two-way pairing: the phone drives the screen, and typed input flows back into the spoken conversation."))

write("seq_outreach.html", figure(seq("outreach", "Owner-directed outreach over the private channel",
    [("ow", "Owner"), ("tg", "telegram.ts"), ("sms", "Twilio SMS"), ("ct", "Contact"), ("ag", "sms.ts agent")],
    [
        ("msg", "ow", "tg", '"text Jane at +1 555… about the role"'),
        ("note", "tg", "gate 1: secret_token header · gate 2: owner id; anyone else is dropped silently"),
        ("ret", "tg", "ow", "confirms name, number, topic"),
        ("msg", "ow", "tg", '"yes, send it"'),
        ("msg", "tg", "sms", "compliance-complete first-contact SMS"),
        ("self", "tg", "outreach row · status = sent"),
        ("msg", "sms", "ct", "delivered from the public number"),
        ("msg", "ct", "ag", "reply text"),
        ("self", "ag", "outreach → replied · thread context injected"),
        ("ret", "ag", "ct", "reply that already knows who they are and why"),
        ("msg", "ow", "tg", '"how’s the Jane thread going?" → status from the ledger'),
    ]),
    "The owner directs; Elle executes and remembers. When the contact replies, the SMS agent already has the context."))

write("seq_escalate.html", figure(seq("escalate", "Roadmap: the director escalation (fuse layers 1+2)",
    [("c", "Caller"), ("tk", "Duplex talker"), ("br2", "Inference brain"), ("tt", "House TTS")],
    [
        ("msg", "c", "tk", "audio: the talker hears AND judges turns natively"),
        ("ret", "tk", "c", "routine turn: answered directly via house TTS"),
        ("msg", "c", "tk", "hard question (pricing, policy, tool work)"),
        ("msg", "tk", "br2", "ask_brain(question), a tool call not a transfer"),
        ("msg", "tk", "tt", "pre-synthesized backchannel: “let me check that”"),
        ("msg", "br2", "tt", "sentence 1 streamed while still generating"),
        ("msg", "tt", "c", "first audio ≈ 1 s, same house voice"),
        ("ret", "br2", "tk", "spoken text fed back as the talker's context"),
        ("note", ("tk", "tt"), "text bottleneck preserved: verbatim compliance wording + swappable voice survive"),
    ]),
    "The future architecture keeps inference and TTS separate on purpose: cost tiering, tool reliability, and verbatim delivery all live at that seam."))

# ---------------------------------------------------------------- ERD
# Crow's-foot notation: underlined primary keys, cardinality glyphs at BOTH
# ends of every relationship (tick = one, circle = zero/optional, three-prong
# foot = many), verb-phrase labels, and a notation legend. Participation is
# derived from the actual schema: nullable correlational keys draw as optional.

def entity(x, y, name, cols, w=196):
    rh = 19
    h = 26 + rh * len(cols)
    p = [f'<g class="erd-ent"><rect class="erd-box" x="{x}" y="{y}" width="{w}" height="{h}" rx="8"/>']
    p.append(f'<rect class="erd-head" x="{x}" y="{y}" width="{w}" height="24" rx="8"/>')
    p.append(f'<rect class="erd-head" x="{x}" y="{y+14}" width="{w}" height="10"/>')
    p.append(f'<text class="erd-title" x="{x+w/2}" y="{y+17}" text-anchor="middle">{esc(name)}</text>')
    for i, (cn, ct, marker) in enumerate(cols):
        cy = y + 26 + rh * i + 13
        prefix = {"PK": "", "FK": "FK ", "": ""}[marker]
        cls = "erd-col erd-pk" if marker == "PK" else ("erd-col erd-fk" if marker == "FK" else "erd-col")
        p.append(f'<text class="{cls}" x="{x+10}" y="{cy}">{esc(prefix + cn)}</text>')
        if marker == "PK":
            # classic ERD convention: the primary key attribute is UNDERLINED
            uw = len(cn) * 6.2
            p.append(f'<line class="erd-underline" x1="{x+10}" y1="{cy+3}" x2="{x+10+uw:.0f}" y2="{cy+3}"/>')
        p.append(f'<text class="erd-type" x="{x+w-10}" y="{cy}" text-anchor="end">{esc(ct)}</text>')
    p.append('</g>')
    return "".join(p), h


def _card_glyph(p, ex, ey, dx, dy, card):
    """Cardinality glyph at endpoint (ex,ey); (dx,dy) is the unit direction
    pointing AWAY from the entity along the line. tick = one, circle = zero,
    fan = many. card ∈ {"1","01","0N","1N"}."""
    nx, ny = -dy, dx  # normal
    def L(a, b, c, d):
        p.append(f'<line class="erd-card" x1="{a:.0f}" y1="{b:.0f}" x2="{c:.0f}" y2="{d:.0f}"/>')
    if card in ("1", "01"):
        t = 10  # tick position along the line
        L(ex + dx*t - nx*5, ey + dy*t - ny*5, ex + dx*t + nx*5, ey + dy*t + ny*5)
    if card in ("0N", "1N"):
        # crow's foot: prongs from a point 12px out, spreading back to the edge
        for off in (-6, 0, 6):
            L(ex + dx*12, ey + dy*12, ex + nx*off, ey + ny*off)
        t = 16
        if card == "1N":
            L(ex + dx*t - nx*5, ey + dy*t - ny*5, ex + dx*t + nx*5, ey + dy*t + ny*5)
    if card in ("01", "0N"):
        c = 22 if card == "0N" else 17
        p.append(f'<circle class="erd-card erd-card-o" cx="{ex + dx*c:.0f}" cy="{ey + dy*c:.0f}" r="4"/>')


ents = {}

def rel(p, a, b, card_a, card_b, verb, ay_frac=0.5, by_frac=0.5, xa_frac=0.5, xb_frac=0.5):
    """Relationship a→b. card_a is a's cardinality seen from b (drawn at a's
    end), card_b likewise at b's end. Horizontal when the boxes don't overlap
    horizontally; vertical otherwise."""
    ax, ay, aw, ah = ents[a]; bx, by_, bw, bh = ents[b]
    if ax + aw <= bx or bx + bw <= ax:  # horizontal
        if ax + aw <= bx:
            x1, x2 = ax + aw, bx
            d1, d2 = (1, 0), (-1, 0)
        else:
            x1, x2 = ax, bx + bw
            d1, d2 = (-1, 0), (1, 0)
        y1 = ay + ah * ay_frac; y2 = by_ + bh * by_frac
        mx = (x1 + x2) / 2
        p.append(f'<path class="erd-rel" d="M {x1} {y1} C {mx} {y1}, {mx} {y2}, {x2} {y2}" fill="none"/>')
        _card_glyph(p, x1, y1, *d1, card_a)
        _card_glyph(p, x2, y2, *d2, card_b)
        lx, ly = mx, min(y1, y2) + abs(y2 - y1) / 2 - 7
    else:  # vertical
        if ay + ah <= by_:
            y1, y2 = ay + ah, by_
            d1, d2 = (0, 1), (0, -1)
        else:
            y1, y2 = ay, by_ + bh
            d1, d2 = (0, -1), (0, 1)
        x1 = ax + aw * xa_frac; x2 = bx + bw * xb_frac
        my = (y1 + y2) / 2
        p.append(f'<path class="erd-rel" d="M {x1} {y1} C {x1} {my}, {x2} {my}, {x2} {y2}" fill="none"/>')
        _card_glyph(p, x1, y1, *d1, card_a)
        _card_glyph(p, x2, y2, *d2, card_b)
        lx, ly = (x1 + x2) / 2 + 6, my + 4
    p.append(f'<text class="sdg-lbl erd-verb" x="{lx:.0f}" y="{ly:.0f}" text-anchor="middle">{esc(verb)}</text>')


p = []
W, H = 900, 774
p.append(f'<svg class="sdg erd" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="Entity relationship diagram of the Elle SQLite schema">')

def put(key, x, y, name, cols, w=196):
    svg, h = entity(x, y, name, cols, w)
    ents[key] = (x, y, w, h)
    p.append(svg)

put("callers", 20, 210, "callers", [
    ("phone_number", "TEXT", "PK"), ("first_seen_at", "TEXT", "")])
put("calls", 350, 20, "calls", [
    ("call_sid", "TEXT", "PK"), ("phone_number", "TEXT", "FK"),
    ("started_at", "TEXT", ""), ("ended_at", "TEXT", "")])
put("facts", 350, 200, "facts", [
    ("id", "INTEGER", "PK"), ("phone_number", "TEXT", "FK"),
    ("category", "TEXT", ""), ("fact", "TEXT", ""),
    ("call_sid", "TEXT", "FK"), ("created_at", "TEXT", "")])
put("transcripts", 680, 20, "transcripts", [
    ("id", "INTEGER", "PK"), ("call_sid", "TEXT", "FK"),
    ("role", "TEXT", ""), ("content", "TEXT", ""), ("created_at", "TEXT", "")])
put("bookings", 350, 400, "bookings", [
    ("id", "INTEGER", "PK"), ("phone_number", "TEXT", "FK"),
    ("call_sid", "TEXT", "FK"), ("name", "TEXT", ""), ("email", "TEXT", ""),
    ("topic", "TEXT", ""), ("start_at", "TEXT", ""), ("status", "TEXT", ""),
    ("event_id / event_link", "TEXT", "")])
put("outreach", 20, 420, "outreach", [
    ("phone_number", "TEXT", "PK"), ("name", "TEXT", ""), ("topic", "TEXT", ""),
    ("message", "TEXT", ""), ("status", "TEXT", "")])
put("sms_consent", 20, 40, "sms_consent", [
    ("phone_number", "TEXT", "PK"), ("status", "TEXT", ""),
    ("source", "TEXT", ""), ("updated_at", "TEXT", "")])
put("settings", 680, 430, "settings", [
    ("key", "TEXT", "PK"), ("value", "TEXT", ""), ("updated_at", "TEXT", "")])

# parent → child, participation from the actual nullable columns
rel(p, "callers", "calls", "1", "0N", "places", 0.28, 0.5)
rel(p, "callers", "facts", "1", "0N", "is described by", 0.62, 0.35)
rel(p, "calls", "transcripts", "1", "0N", "records", 0.5, 0.5)
rel(p, "calls", "facts", "01", "0N", "produces", xa_frac=0.72, xb_frac=0.72)
rel(p, "callers", "bookings", "01", "0N", "requests", 0.86, 0.3)
rel(p, "callers", "sms_consent", "1", "01", "grants texting via", 0.18, 0.6)
rel(p, "callers", "outreach", "1", "01", "is targeted by", 0.95, 0.25)

# notation legend
ly0 = 646
p.append(f'<g class="erd-legendbox"><rect class="erd-box" x="20" y="{ly0-18}" width="{W-40}" height="104" rx="8"/>')
p.append(f'<text class="erd-title" x="{W/2}" y="{ly0}" text-anchor="middle">Notation</text>')
lx = 44
def leg(card, label):
    global lx
    p.append(f'<line class="erd-rel" x1="{lx}" y1="{ly0+22}" x2="{lx+34}" y2="{ly0+22}"/>')
    _card_glyph(p, lx, ly0+22, 1, 0, card)
    p.append(f'<text class="sdg-lbl" x="{lx+42}" y="{ly0+26}" text-anchor="start">{esc(label)}</text>')
    lx += 42 + len(label)*6.2 + 26
leg("1", "exactly one")
leg("01", "zero or one")
leg("1N", "one or many")
leg("0N", "zero or many")
p.append(f'<text class="sdg-lbl" x="44" y="{ly0+48}" text-anchor="start">underlined attribute = primary key · FK = correlational key (convention, not enforced)</text>')
p.append(f'<text class="sdg-lbl" x="44" y="{ly0+64}" text-anchor="start">many-to-many: a foot on both ends, resolved through a junction table.</text>')
p.append(f'<text class="sdg-lbl" x="44" y="{ly0+80}" text-anchor="start">None exists in this schema; every relationship hangs off one person or one call.</text>')
p.append('</g>')
p.append('</svg>')
write("erd.html", figure("".join(p), "The full schema in crow's-foot notation. A caller places calls, calls record transcripts and produce facts, and one-row-per-number tables (consent, outreach) hang directly off the person."))

# ---------------------------------------------------------------- crossover chart
import math
def crossover_chart():
    W, H = 780, 360
    x0, x1, y0, y1 = 80, 730, 300, 40          # plot box (px)
    lx0, lx1 = 3.0, 7.0                        # log10 minutes/month: 1K .. 10M
    ly0, ly1 = 1.0, 5.4                        # log10 $/month: $10 .. $250K
    def X(v): return x0 + (math.log10(v) - lx0) / (lx1 - lx0) * (x1 - x0)
    def Y(d): return y0 - (math.log10(d) - ly0) / (ly1 - ly0) * (y0 - y1)
    p = [f'<svg class="sdg chart" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="Monthly infrastructure cost versus call volume for each architecture, log-log, with crossover points">']
    # gridlines + ticks
    for v, lab in [(1e3,"1K"),(1e4,"10K"),(1e5,"100K"),(1e6,"1M"),(1e7,"10M")]:
        p.append(f'<line class="sdg-life" x1="{X(v):.0f}" y1="{y1}" x2="{X(v):.0f}" y2="{y0}"/>')
        p.append(f'<text class="sdg-lbl" x="{X(v):.0f}" y="{y0+18}" text-anchor="middle">{lab}</text>')
    for d, lab in [(10,"$10"),(100,"$100"),(1e3,"$1K"),(1e4,"$10K"),(1e5,"$100K")]:
        p.append(f'<line class="sdg-life" x1="{x0}" y1="{Y(d):.0f}" x2="{x1}" y2="{Y(d):.0f}"/>')
        p.append(f'<text class="sdg-lbl" x="{x0-8}" y="{Y(d)+4:.0f}" text-anchor="end">{lab}</text>')
    p.append(f'<text class="sdg-lbl" x="{(x0+x1)//2}" y="{y0+34}" text-anchor="middle">talking minutes per month (log)</text>')
    p.append(f'<text class="sdg-lbl" x="18" y="{(y0+y1)//2}" text-anchor="middle" transform="rotate(-90 18 {(y0+y1)//2})">replaceable infra $/month (log)</text>')
    # hosted APIs: $0.0075/min (caching-optimized speech+LLM spend; telephony excluded — common to all)
    pts = " ".join(f"{X(v):.0f},{Y(0.0075*v):.0f}" for v in [1e3, 1e4, 8e4, 1e6, 4.1667e6, 1e7])
    p.append(f'<polyline class="ch-hosted" points="{pts}" fill="none"/>')
    # single self-hosted GPU: ~$600/mo flat to its concurrency ceiling (~300K min/mo)
    p.append(f'<polyline class="ch-gpu" points="{X(1e3):.0f},{Y(600):.0f} {X(3e5):.0f},{Y(600):.0f}" fill="none"/>')
    p.append(f'<line class="ch-gpu ch-cap" x1="{X(3e5):.0f}" y1="{Y(600):.0f}" x2="{X(3e5):.0f}" y2="{y1+16}"/>')
    p.append(f'<text class="sdg-lbl" x="{X(3e5):.0f}" y="{y1+12}" text-anchor="middle">single-GPU ceiling</text>')
    # fleet: ~$31,250/mo fixed + small marginal, drawn from where a fleet is plausible
    pts = " ".join(f"{X(v):.0f},{Y(31250 + 0.0004*v):.0f}" for v in [2e5, 1e6, 4.1667e6, 1e7])
    p.append(f'<polyline class="ch-fleet" points="{pts}" fill="none"/>')
    # crossover markers
    for v, d, lab in [(8e4, 600, "80K min/mo: first GPU pencils"), (4.1667e6, 31250+0.0004*4.1667e6, "4.2M min/mo: fleet wins outright")]:
        p.append(f'<circle class="ch-dot" cx="{X(v):.0f}" cy="{Y(d):.0f}" r="5"/>')
        anchor = "start" if v < 1e6 else "end"
        dx = 10 if v < 1e6 else -10
        p.append(f'<text class="sdg-lbl ch-mark" x="{X(v)+dx:.0f}" y="{Y(d)-10:.0f}" text-anchor="{anchor}">{lab}</text>')
    # legend
    lx, ly = x0 + 8, y1 + 6
    for cls, lab in [("ch-hosted","hosted APIs (cached cascade)"), ("ch-gpu","one self-hosted GPU"), ("ch-fleet","self-hosted fleet")]:
        p.append(f'<line class="{cls}" x1="{lx}" y1="{ly+5}" x2="{lx+26}" y2="{ly+5}"/>')
        p.append(f'<text class="sdg-lbl" x="{lx+32}" y="{ly+9}" text-anchor="start">{lab}</text>')
        ly += 18
    p.append('</svg>')
    return "".join(p)

write("chart_crossover.html", figure(crossover_chart(),
    "Where each architecture is cheapest (log-log, schematic). Below the first dot, hosted APIs "
    "win by an order of magnitude; between the dots, hybrids pencil; past the second, a fleet wins "
    "outright. Telephony (~$0.009/min) is excluded; every option pays it equally."))

# ------------------------------------------------------------------ inject
# The docs pages carry these diagrams INLINE (which is why the docs site needs
# no assets and why the diagrams follow the reader's theme). They used to be
# pasted in by hand, so regenerating here changed nothing anyone could see and
# the pages quietly drifted from the generator. Now the pages are rewritten in
# place: each freshly generated <svg> replaces the one with the same aria-label,
# wherever it appears.
#
# Only the <svg> is replaced, never the surrounding <figure> or <figcaption>: a
# diagram appears both on the collected Diagrams page and in context on its home
# page, and those two carry different captions on purpose.
import re

DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "public", "docs")


def inject():
    fresh = {}
    for fname in sorted(os.listdir(OUT)):
        if not fname.endswith(".html"):
            continue
        content = open(os.path.join(OUT, fname)).read()
        for svg in re.findall(r"<svg\b.*?</svg>", content, re.S):
            label = re.search(r'aria-label="([^"]*)"', svg)
            if label:
                fresh[label.group(1)] = svg

    total = 0
    for page in sorted(os.listdir(DOCS)):
        if not page.endswith(".html"):
            continue
        path = os.path.join(DOCS, page)
        before = open(path).read()

        def swap(m):
            label = re.search(r'aria-label="([^"]*)"', m.group(0))
            if not label or label.group(1) not in fresh:
                return m.group(0)
            return fresh[label.group(1)]

        after = re.sub(r"<svg\b[^>]*\bclass=\"sdg[^\"]*\".*?</svg>", swap, before, flags=re.S)
        if after != before:
            open(path, "w").write(after)
            n = len(re.findall(r"<svg\b[^>]*\bclass=\"sdg", after))
            total += n
            print(f"updated {page} ({n} diagrams)")
    print(f"{total} inlined diagrams refreshed from {len(fresh)} generated")


inject()
