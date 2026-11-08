(() => {
  const pairScreen = document.getElementById("pair-screen");
  const sessionScreen = document.getElementById("session-screen");
  const pairForm = document.getElementById("pair-form");
  const codeInput = document.getElementById("code-input");
  const pairError = document.getElementById("pair-error");
  const stage = document.getElementById("stage");
  const transcriptEl = document.getElementById("transcript");
  const typeBar = document.getElementById("typebar");
  const typeInput = document.getElementById("type-input");
  const statusDot = document.getElementById("status-dot");
  const statusText = document.getElementById("status-text");

  let socket = null;

  pairForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    pairError.textContent = "";
    const code = codeInput.value.replace(/\D/g, "");
    if (code.length !== 6) {
      pairError.textContent = "Please enter the 6-digit code.";
      return;
    }
    try {
      const res = await fetch("/api/pair", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code }),
      });
      const data = await res.json();
      if (!res.ok) {
        pairError.textContent = data.error || "Pairing failed. Try again.";
        return;
      }
      connect(data.token);
    } catch {
      pairError.textContent = "Could not reach the server. Try again.";
    }
  });

  function connect(token) {
    const protocol = location.protocol === "https:" ? "wss" : "ws";
    socket = new WebSocket(`${protocol}://${location.host}/ws?token=${encodeURIComponent(token)}`);

    socket.addEventListener("open", () => {
      pairScreen.hidden = true;
      sessionScreen.hidden = false;
      setStatus(true, "Connected to your call");
    });

    socket.addEventListener("message", (event) => {
      let msg;
      try {
        msg = JSON.parse(event.data);
      } catch {
        return;
      }
      handleEvent(msg);
    });

    socket.addEventListener("close", () => {
      setStatus(false, "Disconnected");
    });
  }

  function setStatus(connected, text) {
    statusDot.classList.toggle("connected", connected);
    statusText.textContent = text;
  }

  function handleEvent(msg) {
    switch (msg.type) {
      case "transcript":
        addTranscript(msg.role, msg.text);
        break;
      case "image":
        showImage(msg.url, msg.caption);
        break;
      case "choices":
        showChoices(msg.question, msg.options);
        break;
      case "call_ended":
        setStatus(false, "Call ended — thanks for talking!");
        break;
    }
  }

  function addTranscript(role, text) {
    const line = document.createElement("div");
    line.className = `line ${role}`;
    const who = document.createElement("span");
    who.className = "who";
    who.textContent = role === "assistant" ? "AI" : "You";
    line.appendChild(who);
    line.appendChild(document.createTextNode(text));
    transcriptEl.appendChild(line);
    transcriptEl.scrollTop = transcriptEl.scrollHeight;
  }

  function clearStage() {
    stage.innerHTML = "";
  }

  function showImage(url, caption) {
    clearStage();
    const img = document.createElement("img");
    img.src = url;
    img.alt = caption || "Image from your call";
    img.addEventListener("error", () => {
      stage.innerHTML = '<p class="placeholder">That image could not be loaded.</p>';
    });
    stage.appendChild(img);
    if (caption) {
      const cap = document.createElement("p");
      cap.className = "caption";
      cap.textContent = caption;
      stage.appendChild(cap);
    }
  }

  function showChoices(question, options) {
    clearStage();
    const q = document.createElement("p");
    q.className = "question";
    q.textContent = question;
    stage.appendChild(q);

    const wrap = document.createElement("div");
    wrap.className = "choices";
    options.forEach((option) => {
      const btn = document.createElement("button");
      btn.textContent = option;
      btn.addEventListener("click", () => {
        if (socket && socket.readyState === WebSocket.OPEN) {
          socket.send(JSON.stringify({ type: "choice", option }));
        }
        wrap.querySelectorAll("button").forEach((b) => (b.disabled = true));
        btn.classList.add("picked");
      });
      wrap.appendChild(btn);
    });
    stage.appendChild(wrap);
  }

  // Typed messages ride the same conversation as speech: the server injects
  // them as a turn, Elle answers out loud, and the transcript (including the
  // typed line itself) streams back over the socket — so we don't render it
  // locally, avoiding duplicates.
  typeBar.addEventListener("submit", (event) => {
    event.preventDefault();
    const text = typeInput.value.trim();
    if (!text) return;
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    socket.send(JSON.stringify({ type: "typed", text }));
    typeInput.value = "";
  });
})();
