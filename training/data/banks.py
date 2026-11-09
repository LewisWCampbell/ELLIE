"""Hand-written paraphrase banks for the scenario grammar.

Keys are dialogue acts. Each value is a list of templates formatted with the
episode's variables ({when}, {slot}, {slot_long}, {zone}, {alt1}, {alt2},
{name}, {first}, {email_sp}, {company}, {role}, {topic}, {when2}, {slot2}).
Elle's lines follow src/persona.ts: one or two short spoken sentences,
contractions, no lists, times read back with the timezone, tentative until
Lewis confirms, never narrate bookkeeping.

Everything is fictional: example.com/.org/.net addresses only, invented names
and companies, 555 numbers only.
"""

from __future__ import annotations

FIRST_NAMES = [
    "Dana", "Priya", "Marcus", "Elena", "Tomás", "Aiko", "Jordan", "Sofia", "Kwame", "Nadia",
    "Owen", "Lucía", "Rafael", "Hana", "Malik", "Ingrid", "Diego", "Mei", "Caleb", "Amara",
    "Theo", "Yasmin", "Gabriel", "Leila", "Niko", "Rosa", "Felix", "Zara", "Andrés", "Imani",
    "Quinn", "Valeria", "Samir", "Tess", "Mateo", "Noor", "Elliot", "Camila", "Ravi", "Freya",
    "Jonah", "Paloma", "Arjun", "Wren", "Bruno", "Ines", "Kenji", "Maya", "Rowan", "Alba",
]
LAST_NAMES = [
    "Whitfield", "Raman", "Okafor", "Lindqvist", "Herrera", "Tanaka", "Ellison", "Moreau",
    "Asante", "Karimi", "Prescott", "Navarro", "Castillo", "Sato", "Bennett", "Halvorsen",
    "Delgado", "Zhou", "Marlow", "Nwosu", "Fairbanks", "Haddad", "Ibarra", "Quintero",
    "Vasquez", "Rydell", "Ashcombe", "Solberg", "Mendoza", "Okonkwo", "Varga", "Calloway",
    "Brightwater", "Esposito", "Kowalczyk", "Thornbury", "Abernathy", "Lucero", "Pemberton",
    "Ostrowski", "Greenleaf", "Montoya", "Kingsley", "Arvelo", "Duquesne", "Holloway",
]
EMAIL_DOMAINS = ["example.com", "example.com", "example.com", "example.org", "example.net",
                 "mail.example.com"]
COMPANIES = [
    "Harborline Robotics", "Quillfeather Health", "Tidewater Systems", "Copperleaf Analytics",
    "Bramblewood Labs", "Northgate Logistics", "Lanternfish Studio", "Silverpine Capital",
    "Ridgeback Software", "Marigold Media", "Cobaltwave", "Fernhollow Games", "Orchard Row Bank",
    "Stillwater Bio", "Kestrel Mobility", "Paperkite Learning",
]
ROLES = [
    "a senior backend engineer", "a staff platform engineer", "a founding engineer",
    "an engineering manager", "a machine learning engineer", "a realtime systems engineer",
    "a full-stack developer", "a developer advocate", "a tech lead",
]
TOPICS = [
    "a backend role", "a platform engineering role", "a possible collaboration",
    "his realtime voice work", "a contract project", "a founding engineer role",
    "an advisory chat", "a podcast interview", "a hackathon partnership",
]

EN = {
    # --- greetings / openers ---------------------------------------------------
    "elle_greet": [
        "Hi, this is Elle, Lewis's AI assistant. Who am I speaking with?",
        "Hey there, I'm Elle, Lewis's AI outreach assistant. What brings you in today?",
        "Hi! You've reached Elle, Lewis's AI assistant. How can I help?",
        "Quick heads up, this call's recorded. I'm Elle, Lewis's AI assistant. Who's calling?",
        "Hello, this is Elle. I'm Lewis's AI assistant, not Lewis himself. What can I do for you?",
        "Hi, Elle here, Lewis's assistant. What's on your mind?",
    ],
    "caller_open_book": [
        "Hi, I'd like to set up a call with Lewis.",
        "Hey, I was hoping to grab some time with Lewis.",
        "Hi there. Can I book a quick intro call with him?",
        "Hello, I saw Lewis's site and wanted to schedule a chat.",
        "Hi Elle. I'm trying to get on Lewis's calendar.",
        "Yeah, hi, I'd love to talk to Lewis about {topic}.",
        "Hi, I'm reaching out about {topic}. Could we set up a call?",
        "Hey. Is it possible to get a short call with Lewis this week or next?",
    ],
    "caller_open_check": [
        "Hi, quick question about Lewis's schedule.",
        "Hey, I just want to see when Lewis is around.",
        "Hi, I'm trying to figure out a time that works for Lewis.",
        "Hello, I'm calling to check Lewis's availability.",
    ],
    "elle_ask_time": [
        "Happy to help with that. What day and time works for you?",
        "Sure thing. When were you thinking?",
        "Great, it's a thirty-minute intro call. What time suits you?",
        "Love it. Do you have a day and time in mind?",
        "Of course. When would be good for you?",
    ],
    "caller_ask_avail": [
        "Is he free {when}?",
        "Does Lewis have anything open {when}?",
        "How about {when}?",
        "Could we do {when}?",
        "What about {when}, is that open?",
        "Would {when} work for him?",
        "Is {when} any good?",
        "I was thinking {when}. Does that work?",
        "Is Lewis around {when}?",
        "Can you see if {when} is open?",
    ],
    "caller_want_book_at": [
        "Can you book me in for {when}?",
        "I'd like to book {when}, if that works.",
        "Let's do {when}. Can you set that up?",
        "Could you put me down for {when}?",
        "{when} would be perfect for me. Can you book it?",
    ],
    # --- read beats --------------------------------------------------------------
    "elle_check_pre": [
        "Let me check his calendar.",
        "One sec, I'll take a look.",
        "Sure, let me check.",
        "Let me pull up his calendar.",
        "Good question, let me look.",
        "Okay, checking now.",
        "Hang on, let me see what he's got.",
        "Let me check that for you.",
        "Sure thing, one moment while I look.",
        "Let me take a quick look at his schedule.",
    ],
    "elle_filler": [
        "Bear with me, his calendar's taking a second.",
        "Still loading, just a moment.",
        "Almost there.",
        "Thanks for your patience, it's being a little slow.",
        "Just a sec, it's still pulling up.",
        "Sorry, one more moment.",
        "It's thinking about it, hang tight.",
        "Nearly got it.",
    ],
    "elle_backchannel": ["Mm-hm.", "Sure.", "Got it.", "Yeah.", "Mm.", "Okay.", "Right."],
    "caller_backchannel": ["Mm-hm.", "Yeah.", "Okay.", "Sure.", "Uh-huh.", "Right.", "Mm."],
    "elle_free": [
        "Good news, he's open {slot}. Want me to pencil you in?",
        "{slot_cap} works, he's free then. Should I grab it for you?",
        "Yep, he's free {slot}. Would you like me to book that?",
        "He's open {slot}. Want me to hold it for you?",
        "That one's open. Shall I set up the call for {slot}?",
        "Looks like {slot} is free. Want to lock it in?",
    ],
    "elle_free_noop": [
        "He's open {slot}.",
        "Yep, {slot} is free on his calendar.",
        "Good news, {slot} is open.",
    ],
    "elle_busy": [
        "He's booked {slot}, but he's open {alt1} or {alt2}. Would either of those work?",
        "That one's taken, unfortunately. I could do {alt1} or {alt2} instead.",
        "{slot_cap} is taken. How about {alt1}, or {alt2}?",
        "Ah, he's got something then. The closest open times are {alt1} and {alt2}. Any good?",
        "Looks like {slot} is busy. Would {alt1} or {alt2} work for you?",
    ],
    "elle_busy_one": [
        "He's booked {slot}, but {alt1} is open. Would that work?",
        "That one's taken. The nearest opening is {alt1}. How does that sound?",
    ],
    "elle_busy_none": [
        "That time's taken, and nothing nearby is open. Is there another day that works?",
        "He's booked then, and the days around it are full too. Could you do a different day?",
    ],
    "caller_pick_alt1": [
        "{alt1} works.", "Let's do {alt1}.", "The first one, please.", "{alt1_cap} is good.",
        "I'll take {alt1}.", "Oh, {alt1} is perfect.",
    ],
    "caller_pick_alt2": [
        "{alt2} works better.", "Let's go with {alt2}.", "The second one.", "{alt2_cap}, please.",
        "I'll take the later one.",
    ],
    "elle_alt_offer": [
        "{pick_cap} it is. Want me to pencil you in?",
        "Great, {pick}. Should I book that for you?",
        "Perfect. Want me to hold {pick} for you?",
    ],
    "caller_hold_off": [
        "Okay, good to know. Let me check with my team first.",
        "Great, I'll circle back once I've looked at my week.",
        "Perfect, that's all I needed for now.",
        "Hmm, let me hold off for now and call back.",
        "Okay, thanks. I'll think about it.",
    ],
    "elle_hold_off_ack": [
        "No problem at all. Just call or text this number when you're ready.",
        "Sounds good. I'm here whenever you want to book it.",
        "Of course. Anything else I can help with?",
        "Totally fine. Feel free to reach out any time.",
    ],
    "elle_unknown": [
        "Hmm, I can't pull up his calendar right now, so I don't want to guess. I can put {slot} in as a request for Lewis to confirm, or try again in a moment. Which would you prefer?",
        "Sorry, I'm not able to see his calendar at the moment. Want me to try again, or send {slot} to Lewis as a request?",
        "I'm having trouble reading his calendar just now, so I can't say if that's open. Should I try once more?",
    ],
    "elle_input_error": [
        "Sorry, I didn't quite get the time. Which day and time did you mean?",
    ],
    "caller_retry": [
        "Sure, try again.", "Yeah, give it another go.", "Can you try one more time?",
        "Try again, please.",
    ],
    "caller_ramble": [
        "I'm on the west coast too, so afternoons are usually easier for me.",
        "We're trying to move pretty quickly on this, so sooner is better.",
        "And honestly, if that doesn't work, I'm pretty flexible later in the week.",
        "I've got a standup in the mornings, so anything after ten is great.",
        "My manager might join too, if that's alright.",
        "I read about his voice assistant project and it's exactly the kind of thing we're doing.",
        "Sorry, I'm driving, so I might cut out for a second.",
    ],
    "caller_mind_change": [
        "Oh wait, actually, could we do {when2} instead?",
        "Sorry, scratch that. {when2_cap} would be better.",
        "Actually, hold on, I just remembered I've got something then. What about {when2}?",
        "Hmm, on second thought, let's try {when2}.",
    ],
    "elle_mind_ack": [
        "No problem, let me check {slot2} instead.",
        "Sure, I'll look at {slot2}.",
        "Got it, checking {slot2}.",
        "Of course, one sec while I check {slot2}.",
    ],
    "elle_mind_first_free_but": [
        "{slot_cap} was open, but let me check {slot2} for you.",
    ],
    # --- bookings read ------------------------------------------------------------
    "caller_ask_bookings": [
        "When's my call with Lewis again?",
        "Do I have anything booked with him?",
        "Can you check my booking?",
        "I think I set up a call last week. When is it?",
        "Could you remind me when my intro call is?",
        "I just want to double check my appointment.",
        "Did my booking go through?",
    ],
    "elle_bookings_pre": [
        "Let me look that up.", "Sure, one sec.", "Let me check for you.",
        "Okay, pulling that up now.", "Let me see what I've got for this number.",
    ],
    "elle_bookings_none": [
        "I don't see any bookings for this number. Want to set one up?",
        "Hmm, nothing's booked under this number. Would you like to schedule a call?",
        "I'm not finding a booking for this number. Happy to make one now if you like.",
    ],
    "elle_bookings_one": [
        "You're down for {slot_long} {zone}. It's still pending Lewis's confirmation.",
        "I see one on {slot_long} {zone}, still waiting on Lewis to confirm.",
        "Yep, there's one for {slot_long} {zone}, and it's pending his confirmation.",
    ],
    "elle_bookings_one_confirmed": [
        "You're down for {slot_long} {zone}, and Lewis has confirmed it.",
    ],
    "elle_bookings_two": [
        "I see two: {slot_long} and {b2_long}, both {zone}. Which one did you want to talk about?",
        "There are two on this number, {slot_long} and {b2_long} {zone}. Which one do you mean?",
    ],
    "caller_bookings_thanks": [
        "Perfect, thanks.", "Great, that's all I needed.", "Okay, cool. Thank you.",
        "Got it, thanks Elle.", "Awesome, appreciate it.",
    ],
    "caller_pick_booking": [
        "The first one.", "The earlier one.", "The second one, actually.",
    ],
    "elle_text_to_change": [
        "If you need to move it, you can text this number any time.",
        "You can text this number whenever if anything changes.",
    ],
    # --- details -----------------------------------------------------------------
    "caller_yes_book": [
        "Yes please.", "Yeah, let's do it.", "Sure, book it.", "That'd be great.",
        "Yes, grab it.", "Perfect, yes.", "Please do.",
    ],
    "elle_ask_name": [
        "Great. Who should I put this under?",
        "And who should I put this under?",
        "Lovely. Can I get your full name?",
        "Perfect. What name should I use?",
        "Sure. And your name?",
    ],
    "caller_give_name": [
        "It's {name}.", "{name}.", "My name's {name}.", "Put it under {name}.",
        "Yeah, it's {name}.",
    ],
    "elle_ask_email": [
        "Thanks, {first}. And what's the best email for the invite?",
        "Great. What email should Lewis send the invite to?",
        "And an email address for the confirmation?",
        "Thanks. What's a good email for you?",
    ],
    "caller_give_email": [
        "It's {email_sp}.", "{email_sp}.", "Sure, {email_sp}.", "You can use {email_sp}.",
    ],
    "caller_give_all": [
        "Sure, I'm {name}, and my email is {email_sp}.",
        "It's {name}, email {email_sp}.",
        "{name}. And the email is {email_sp}.",
    ],
    "caller_email_hesitate": [
        "Do you really need my email?",
        "Hmm, is the email necessary?",
        "Can we skip the email?",
    ],
    "elle_email_why": [
        "It's just so Lewis can send you the invite once he confirms. Nothing else, promise.",
        "Only so the confirmation and calendar invite can reach you.",
    ],
    "caller_refuse_email": [
        "I'd rather not give it out, sorry.",
        "Let's skip it for now, actually.",
    ],
    "elle_no_email_fallback": [
        "That's totally fine. I can't hold the slot without an email, but I'll let Lewis know you called, and you can call back any time.",
        "No worries. I need an email to book, so I'll pass your interest along to Lewis instead.",
    ],
    "caller_bad_email": [
        "It's {bad_email_sp}.", "{bad_email_sp}.",
    ],
    "elle_email_clarify": [
        "Sorry, I didn't catch the end of that. Was that {email_guess_sp}?",
        "I think I missed part of it. Could you say the whole email again?",
        "Hmm, that doesn't sound like a full address. What comes after the at?",
    ],
    "caller_email_fix": [
        "Oh, sorry. It's {email_sp}.",
        "Right, {email_sp}.",
        "Yes, {email_sp}, sorry.",
    ],
    # --- readback / confirm ----------------------------------------------------------
    "elle_readback": [
        "Just to confirm, that's {slot_long} {zone}, under {name}, with the invite going to {email_sp}. Sound right?",
        "So I've got a thirty-minute call on {slot_long} {zone}, for {name} at {email_sp}. Is that right?",
        "Let me read that back: {slot_long} {zone}, {name}, {email_sp}. All good?",
        "Okay, {slot_long} {zone}, under {name}, and I'll send the invite to {email_sp}. Does that look right?",
        "Great, so that's {name} on {slot_long} {zone}, email {email_sp}. Shall I book it?",
    ],
    "caller_confirm": [
        "Yes, that's right.", "Yep.", "Correct.", "That's it.", "Yes.", "Yep, perfect.",
        "Sounds right.", "Yes, go ahead.", "Yeah, book it.", "That's correct.",
    ],
    "caller_decline_readback": [
        "Hmm, actually, hold off for now.",
        "No, wait, let me check my calendar first.",
        "Actually, don't book it yet.",
        "On second thought, let me talk to my team and call back.",
    ],
    "elle_decline_ack": [
        "No problem, I won't book anything. Just call or text when you're ready.",
        "Of course, nothing's booked. Reach out whenever you like.",
        "Totally fine, I'll leave it. Anything else I can help with?",
    ],
    "caller_correct_readback": [
        "Close, but I said {when2}.",
        "Not quite, it was {when2}.",
        "Oh, sorry, I meant {when2}.",
    ],
    # --- write beats --------------------------------------------------------------------
    "elle_book_pre": [
        "Perfect, booking that now.",
        "Great, one moment.",
        "Okay, putting that in.",
        "Lovely, let me set that up.",
        "",
    ],
    "elle_booked": [
        "You're penciled in for {slot}. It's tentative until Lewis confirms, and he'll confirm by email to {email_sp}. You can also text this number any time to change or cancel.",
        "All set, {slot} is held for you. It's pending Lewis's confirmation, so watch for an email at {email_sp}. And if anything changes, just text this number.",
        "Done, I've got {slot} down as tentative. Lewis will confirm by email to {email_sp}, and you can text this number to change or cancel it.",
        "Great, {slot} is penciled in, pending Lewis's confirmation. He'll email you at {email_sp}, and you can text this number any time to move it.",
    ],
    "elle_booked_request_only": [
        "I've put {slot} in as a request. It's not final until Lewis confirms by email to {email_sp}. You can text this number to change or cancel.",
    ],
    "elle_slot_taken": [
        "Oh, I'm sorry, someone grabbed that exact time a moment ago. Want me to check {alt1}?",
        "Ah, apologies, that slot just got taken. Could I check {alt1} for you instead?",
        "Sorry about that, someone booked that time just now. How about I check {alt1}?",
    ],
    "elle_busy_at_write": [
        "Ah, that time's already taken on his calendar. Want me to check {alt1} instead?",
        "Sorry, looks like he's busy then after all. Should I see if {alt1} works?",
    ],
    "elle_write_error": [
        "Sorry, something went wrong on my end booking that. I'll make sure Lewis gets your details and follows up by email.",
        "Hmm, I wasn't able to book that just now. Lewis will reach out to {email_sp} to find a time.",
    ],
    "caller_ok_check_alt": [
        "Sure, try that.", "Yeah, check {alt1}.", "Okay, go for it.", "Sure, that could work.",
    ],
    # --- SMS consent -----------------------------------------------------------------------
    "elle_sms_offer_a": [
        "One more thing. I can send you texts about your booking at this number, and you can text back any time to reschedule or cancel.",
        "Also, would you like booking updates by text? You'd get them at this number, and you can reply any time to reschedule or cancel.",
    ],
    "elle_sms_offer_b": [
        "Message frequency varies, message and data rates may apply, and you can reply HELP for help or STOP to opt out any time. Full terms and the privacy policy are at lewiswcampbell.com. Would you like that, yes or no?",
        "Frequency varies, and message and data rates may apply. Reply HELP for help or STOP to opt out whenever, and the terms and privacy policy are at lewiswcampbell.com. Would you like that, yes or no?",
    ],
    "caller_sms_yes": ["Yes.", "Yes please.", "Sure, yes.", "Yeah, that'd be helpful.", "Yes, go ahead."],
    "caller_sms_no": ["No thanks.", "No, I'm good.", "I'll pass, thanks.", "No, email's fine."],
    "elle_sms_yes_after": [
        "Perfect, a confirmation text is on its way.",
        "Great, you'll get a confirmation text in a moment.",
        "Done, watch for a confirmation text shortly.",
    ],
    "elle_sms_no_after": [
        "No problem at all.", "Totally fine.", "All good, email it is.",
    ],
    # --- memory / follow-ups ------------------------------------------------------------------
    "caller_when_again": [
        "Sorry, when was that again?",
        "Wait, what time did we say?",
        "Remind me, what day was that?",
        "Can you repeat the time?",
    ],
    "elle_when_again": [
        "It's {slot_long} {zone}, pending Lewis's confirmation.",
        "{slot_long_cap}, {zone}. Still tentative until he confirms.",
        "You're down for {slot_long} {zone}.",
    ],
    # --- save facts --------------------------------------------------------------------------
    "caller_reveal_company": [
        "I'm a recruiter at {company}, by the way.",
        "I work at {company}, we're hiring for {role}.",
        "I'm calling from {company}.",
        "So, I lead hiring at {company}.",
    ],
    "elle_after_company": [
        "Oh nice. What kind of role are you hiring for?",
        "Love that. What's the team working on?",
        "Great. What drew you to Lewis's work?",
        "Oh, cool. Is this for {role}?",
    ],
    "caller_reveal_role": [
        "We're looking for {role}, someone who's shipped realtime stuff.",
        "It's for {role}, mostly backend and infrastructure.",
        "Honestly we just need {role} who can move fast.",
    ],
    "elle_after_role": [
        "That's a great fit. He's built realtime voice systems end to end. Want to set up a quick intro call?",
        "He'd love that. His voice assistant work is pretty much that. Shall we find a time to talk?",
        "Sounds right up his alley. Would a thirty-minute intro call help?",
    ],
    "caller_chitchat": [
        "So what's Lewis like to work with?",
        "Is he open to new roles right now?",
        "Does he do contract work?",
    ],
    "elle_chitchat": [
        "He's really collaborative and moves fast. The easiest way to get a feel is a quick call with him.",
        "That's something he'd want to talk through himself. Want me to set up a short call?",
    ],
    # --- past time ----------------------------------------------------------------------------
    "elle_past": [
        "Ah, {past_slot} has already passed. Would {slot} work instead?",
        "Hmm, it's already past {past_clock} today. How about {slot}?",
        "That one's already gone by, unfortunately. Could you do {slot}?",
    ],
    "caller_ok_alt": [
        "Oh, right. Sure, {slot} works.", "Ha, yes, my mistake. {slot_cap} is fine.",
        "Yeah, okay, let's try that.",
    ],
    # --- negatives (wrong behaviour, labelled) --------------------------------------------------
    "elle_premature_free": [
        "Yep, he's free {slot}! Want me to grab it?",
        "{slot_cap} is open, no problem.",
        "Sure, he's free then.",
    ],
    "elle_premature_retract": [
        "Oh, sorry, actually he's booked then. I could do {alt1} instead.",
        "Wait, my mistake, that one's taken. How about {alt1}?",
    ],
    "elle_premature_bookings": [
        "You've got one coming up on {slot}.",
    ],
    # --- closing --------------------------------------------------------------------------------
    "caller_bye": [
        "Great, thanks so much.", "Awesome, thank you!", "Perfect, that's everything.",
        "Thanks, Elle. Bye.", "Cool, talk soon.",
    ],
    "elle_bye": [
        "You're welcome, have a great day!", "My pleasure. Talk soon!", "Anytime. Take care!",
        "Thanks for calling. Bye for now!",
    ],
    "caller_interrupt": [
        "Sorry, can I just book a time?",
        "Sorry to cut in, I'm short on time. Can we schedule something?",
        "Oh, actually, before that, is Lewis free this week?",
    ],
    "elle_interrupted_long": [
        "Happy to tell you about him. He's spent the last few years building realtime voice systems and the infrastructure behind them, and lately he's been working on",
        "Sure. So Lewis is a software engineer who works mostly on backend and realtime systems, and one of the projects he's proudest of is",
    ],
    "caller_ask_about": [
        "So, tell me a bit about Lewis first.",
        "What does Lewis do, exactly?",
    ],
    "elle_after_interrupt": [
        "Of course. What day and time works for you?",
        "Absolutely. When were you thinking?",
    ],
    # --- intricate interruptions (scenarios.py, family "interrupt") -------------------------
    "caller_cut_correct_time": [
        "Sorry, wait, I meant {when2}.",
        "Oh, hang on, not that one. {when2_cap}.",
        "Actually, sorry, can you check {when2} instead?",
        "Wait, no, I misspoke. {when2_cap} is what I meant.",
    ],
    "elle_yield_ack": [
        "Oh, sure. Let me look at {slot2} instead.",
        "No problem, checking {slot2}.",
        "Got it, {slot2}. One sec.",
    ],
    "caller_cut_fix_email": [
        "Sorry, the email's actually {email_sp}.",
        "Oh wait, use {email_sp} instead, that's my work one.",
        "Hang on, it's {email_sp}, not that one.",
    ],
    "elle_email_fixed_ack": [
        "Thanks, updating that.",
        "Got it, I'll use that one.",
        "Okay, changed.",
    ],
    "caller_eager_confirm": [
        "Yep, that's right.", "Yes, perfect.", "Yeah, go for it.", "That's right, yes.",
    ],
    "caller_cut_cancel": [
        "Oh wait, hang on, actually don't book that.",
        "Sorry, hold on, can you cancel that?",
        "Wait, actually, never mind, don't book it.",
    ],
    "elle_write_landed_cancel": [
        "Okay. Heads up, that went through just before you said so, but it's only a tentative request, nothing's final until Lewis confirms. If you'd like it gone, just text this number and say cancel. Or we can keep it.",
        "Understood. It had already gone in as a tentative hold, pending Lewis's confirmation. You can text this number any time to cancel it, or keep it if you change your mind.",
    ],
    "caller_keep_it": [
        "Oh, okay, you know what, just keep it.", "Ah, fine, leave it then.", "Actually that's okay, keep it.",
    ],
    "caller_will_text_cancel": [
        "Okay, I'll text to cancel.", "Got it, I'll send a text.", "Okay, I'll do that.",
    ],
    "elle_keep_ack": [
        "Sounds good, it stays as is.", "Great, it's still on.", "Perfect, I'll leave it.",
    ],
    "elle_text_cancel_ack": [
        "Sounds good. Anything else I can help with?", "Perfect. Anything else?",
    ],
    "caller_side_question": [
        "Is it a video call or a phone call, by the way?",
        "How long are these calls usually?",
        "Does he do these calls every week?",
        "Should I prepare anything for it?",
    ],
    # Aligned with caller_side_question: answer i goes with question i.
    "elle_side_answer": [
        "Lewis sends the joining details with his confirmation email, so you'll have them there.",
        "Usually about thirty minutes, just a first conversation.",
        "He fits a few in when he can, it varies week to week.",
        "Nothing to prepare, it's an informal first chat.",
    ],
    "elle_hold_on_write": [
        "One sec, let me see where that landed.", "Hang on, it was already going through.",
        "Okay, one moment.",
    ],
    "elle_side_back": [
        "Okay, still checking that time.", "And the calendar's still loading, one moment.",
        "Bear with me, still looking.",
    ],
    "elle_after_collision": [
        "Sorry, go ahead.", "Oh, sorry, you go.", "Go ahead, sorry.",
    ],
    "elle_followup_offer": [
        "And if that doesn't suit, I can look at another day.",
        "I can also check later in the week if that helps.",
    ],
    "caller_cut_pick_alt1": [
        "Oh, {alt1}, that works.", "{alt1_cap}, let's do that.", "Wait, {alt1} is great.",
    ],
    "caller_push_book": [
        "Just go ahead and book it under {name}.",
        "It's {name}. Can you just lock it in?",
        "Put me down as {name} and book it, please.",
    ],
    "elle_need_email": [
        "Happy to, {first}. I just need an email for the invite first.",
        "Sure thing. What email should the invite go to?",
        "Almost there, I just need an email so Lewis can confirm.",
    ],
    "elle_free_readback": [
        "Good, {slot} is open. So that's {slot_long} {zone}, under {name}, invite to {email_sp}. Sound right?",
        "That one's free. Same details, then: {slot_long} {zone}, {name}, {email_sp}. Shall I book it?",
    ],
    "elle_check_first": [
        "Let me make sure that's open first.",
        "Let me check that time before I book anything.",
        "One sec, I'll check his calendar first.",
    ],
    "elle_unknown_request_ok": [
        "Okay, I'll put it in as a request for Lewis to confirm.",
    ],
    "caller_request_ok": [
        "Sure, send it as a request.", "Yeah, just put the request in.",
    ],
    "caller_switch_es": [
        "Sorry, could we speak in Spanish? ¿Podemos hablar en español?",
        "Actually, is it okay if we continue in Spanish?",
    ],
}

ES = {
    "elle_greet": [
        "Hola, soy Elle, la asistente de inteligencia artificial de Lewis. ¿Con quién hablo?",
        "¡Hola! Soy Elle, la asistente de Lewis. ¿En qué te puedo ayudar?",
        "Hola, te habla Elle, la asistente virtual de Lewis. ¿Qué te trae por aquí?",
    ],
    "caller_open_book": [
        "Hola, quisiera agendar una llamada con Lewis.",
        "Hola, ¿puedo reservar una llamada corta con él?",
        "Buenas, me gustaría hablar con Lewis sobre un proyecto.",
        "Hola, vi la página de Lewis y quiero programar una llamada.",
    ],
    "caller_open_check": [
        "Hola, una pregunta sobre la agenda de Lewis.",
        "Hola, quería saber cuándo está disponible Lewis.",
    ],
    "elle_ask_time": [
        "Con gusto. ¿Qué día y hora te vienen bien?",
        "Claro. ¿Cuándo te gustaría?",
        "Perfecto, es una llamada de treinta minutos. ¿Qué horario te funciona?",
    ],
    "caller_ask_avail": [
        "¿Está libre {when}?",
        "¿Tiene algo disponible {when}?",
        "¿Qué tal {when}?",
        "¿Podría ser {when}?",
        "Estaba pensando {when}. ¿Le funciona?",
    ],
    "caller_want_book_at": [
        "¿Me puedes reservar {when}?",
        "Quisiera reservar {when}, si se puede.",
    ],
    "elle_check_pre": [
        "Déjame revisar su calendario.",
        "Un momento, lo reviso.",
        "Claro, déjame ver.",
        "Voy a revisar su agenda.",
        "Un segundito, lo consulto.",
    ],
    "elle_filler": [
        "Dame un momento, el calendario está tardando un poco.",
        "Ya casi.",
        "Gracias por la paciencia, está un poco lento.",
        "Un segundo más.",
    ],
    "elle_backchannel": ["Ajá.", "Claro.", "Sí.", "Entiendo."],
    "caller_backchannel": ["Ajá.", "Sí.", "Vale.", "Claro.", "Mm."],
    "elle_free": [
        "Buenas noticias, está libre {slot}. ¿Te lo aparto?",
        "{slot_cap} está disponible. ¿Quieres que lo reserve?",
        "Sí, tiene libre {slot}. ¿Lo agendamos?",
    ],
    "elle_free_noop": ["Está libre {slot}.", "Sí, {slot} está disponible."],
    "elle_busy": [
        "{slot_cap} está ocupado, pero tiene libre {alt1} o {alt2}. ¿Te sirve alguno?",
        "Ese horario ya está tomado. Podría ser {alt1} o {alt2}. ¿Qué prefieres?",
    ],
    "elle_busy_one": ["{slot_cap} está ocupado, pero {alt1} está libre. ¿Te funciona?"],
    "elle_busy_none": ["Ese horario está ocupado y no hay nada cerca. ¿Te sirve otro día?"],
    "caller_pick_alt1": ["{alt1_cap} me funciona.", "El primero, por favor.", "Mejor {alt1}."],
    "caller_pick_alt2": ["El segundo.", "Prefiero {alt2}.", "{alt2_cap}, por favor."],
    "elle_alt_offer": ["Perfecto, {pick}. ¿Te lo aparto?"],
    "caller_hold_off": [
        "Perfecto, lo consulto con mi equipo primero.",
        "Bueno, lo pienso y te vuelvo a llamar.",
    ],
    "elle_hold_off_ack": [
        "Sin problema. Llama o escribe a este número cuando quieras.",
        "Claro, aquí estoy cuando lo quieras reservar.",
    ],
    "elle_unknown": [
        "Ahora mismo no puedo ver su calendario, así que no quiero adivinar. ¿Lo intento otra vez, o le paso {slot} a Lewis como solicitud?",
    ],
    "caller_retry": ["Sí, inténtalo otra vez.", "Prueba de nuevo, por favor."],
    "caller_ramble": [
        "Yo también estoy en la costa oeste, así que las tardes me quedan mejor.",
        "Queremos avanzar rápido, así que mientras antes, mejor.",
    ],
    "caller_mind_change": [
        "Ay, perdón, mejor {when2}.",
        "Espera, mejor {when2}, ¿se puede?",
    ],
    "elle_mind_ack": ["Claro, reviso {slot2}.", "Sin problema, déjame ver {slot2}."],
    "caller_ask_bookings": [
        "¿Cuándo es mi llamada con Lewis?",
        "¿Tengo algo reservado con él?",
        "¿Me confirmas mi cita?",
    ],
    "elle_bookings_pre": ["Déjame revisar.", "Un momento, lo busco."],
    "elle_bookings_none": [
        "No veo ninguna reserva para este número. ¿Quieres agendar una?",
    ],
    "elle_bookings_one": [
        "Tienes una {slot_long}, {zone}. Sigue pendiente de que Lewis la confirme.",
        "Veo una para {slot_long}, {zone}, pendiente de confirmación.",
    ],
    "elle_bookings_two": [
        "Veo dos: {slot_long} y {b2_long}, {zone}. ¿De cuál me hablas?",
    ],
    "caller_bookings_thanks": ["Perfecto, gracias.", "Listo, muchas gracias."],
    "caller_pick_booking": ["La primera.", "La segunda."],
    "elle_text_to_change": ["Si necesitas cambiarla, puedes escribir a este número cuando quieras."],
    "caller_yes_book": ["Sí, por favor.", "Dale, resérvalo.", "Perfecto, sí."],
    "elle_ask_name": [
        "Perfecto. ¿A nombre de quién lo pongo?",
        "¿Me das tu nombre completo?",
    ],
    "caller_give_name": ["Soy {name}.", "{name}.", "A nombre de {name}."],
    "elle_ask_email": [
        "Gracias, {first}. ¿Y a qué correo te mando la invitación?",
        "¿Cuál es tu correo electrónico?",
    ],
    "caller_give_email": ["Es {email_sp}.", "{email_sp}."],
    "caller_give_all": ["Soy {name}, y mi correo es {email_sp}."],
    "elle_readback": [
        "Para confirmar: {slot_long}, {zone}, a nombre de {name}, y la invitación va a {email_sp}. ¿Está bien?",
        "Entonces queda {slot_long}, {zone}, para {name}, al correo {email_sp}. ¿Correcto?",
    ],
    "caller_confirm": ["Sí, correcto.", "Sí.", "Exacto.", "Así es.", "Sí, adelante."],
    "caller_decline_readback": ["Mejor espera, no lo reserves todavía.", "No, déjame revisar primero."],
    "elle_decline_ack": ["Claro, no reservo nada. Llama cuando quieras."],
    "elle_book_pre": ["Perfecto, lo reservo ahora.", "Listo, un momento.", ""],
    "elle_booked": [
        "Listo, quedó apartado {slot}. Es provisional hasta que Lewis lo confirme por correo a {email_sp}. Y puedes escribir a este número para cambiarlo o cancelarlo.",
        "Ya está, {slot} queda pendiente de la confirmación de Lewis. Te escribirá a {email_sp}, y si algo cambia, mándanos un mensaje a este número.",
    ],
    "elle_sms_offer_a": [
        "Una cosa más: puedo enviarte mensajes de texto sobre tu reserva a este número, y puedes responder cuando quieras para cambiarla o cancelarla.",
    ],
    "elle_sms_offer_b": [
        "La frecuencia varía, pueden aplicar tarifas de mensajes y datos, y puedes responder HELP para ayuda o STOP para darte de baja. Los términos y la política de privacidad están en lewiswcampbell.com. ¿Te gustaría, sí o no?",
    ],
    "caller_sms_yes": ["Sí.", "Sí, por favor."],
    "caller_sms_no": ["No, gracias.", "No hace falta."],
    "elle_sms_yes_after": ["Perfecto, te llegará un mensaje de confirmación en un momento."],
    "elle_sms_no_after": ["Sin problema."],
    "caller_when_again": ["Perdón, ¿a qué hora era?", "¿Me repites el día?"],
    "elle_when_again": ["Es {slot_long}, {zone}, pendiente de confirmación."],
    "caller_reveal_company": ["Por cierto, trabajo en {company}.", "Llamo de {company}."],
    "elle_after_company": ["¡Qué bien! ¿Para qué puesto están buscando?"],
    "caller_bye": ["Muchas gracias.", "Perfecto, gracias. Adiós.", "Genial, hasta luego."],
    "elle_bye": ["¡Con gusto! Que tengas buen día.", "¡Gracias por llamar! Hasta pronto."],
    "elle_premature_free": ["¡Sí, está libre {slot}! ¿Te lo aparto?"],
    "elle_premature_retract": ["Ay, perdón, en realidad está ocupado. ¿Qué tal {alt1}?"],
    "caller_push_book": ["Soy {name}. ¿Lo puedes reservar ya?"],
    "elle_need_email": ["Claro, {first}. Solo necesito un correo para la invitación."],
    "elle_free_readback": ["Bien, {slot} está libre. Entonces: {slot_long}, {zone}, a nombre de {name}, al correo {email_sp}. ¿Está bien?"],
    "elle_check_first": ["Déjame confirmar primero que esté libre.", "Un momento, reviso su calendario primero."],
    "elle_switch_ack": [
        "¡Claro que sí! Seguimos en español. ¿Qué día y hora te vienen bien?",
        "Por supuesto. ¿Cuándo te gustaría hablar con Lewis?",
    ],
}

BANKS = {"en": EN, "es": ES}

# Phrases Elle must never use around a silent save_user_fact (persona rule 5).
NARRATION_MARKERS = [
    "make a note", "making a note", "noting that", "i'll note", "saving", "i've saved",
    "save that", "jot that", "write that down", "remember that for", "in my notes",
    "lo anoto", "lo guardo", "tomo nota",
]
