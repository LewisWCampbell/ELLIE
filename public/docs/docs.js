// Elle docs: one nav definition shared by every page. Each page includes
// <nav class="sidebar"></nav> + <main> and this script fills the nav in,
// so adding a page means adding one line here.
(function () {
  "use strict";

  var PAGES = [
    ["index.html", "Overview"],
    ["architecture.html", "Architecture"],
    ["diagrams.html", "Diagrams"],
    ["voice-pipeline.html", "The voice pipeline"],
    ["languages.html", "Bilingual mode"],
    ["channels.html", "Web, SMS and intake"],
    ["booking.html", "Booking and compliance"],
    ["data.html", "Data and knowledge"],
    ["database.html", "The database"],
    ["security.html", "Security"],
    ["compliance.html", "Compliance and BAAs"],
    ["cost.html", "Cost engineering"],
    ["scaling.html", "Architectures by volume"],
    ["operations.html", "Operations"],
    ["reference.html", "Code reference"],
    ["features.html", "Feature index"],
    ["glossary.html", "Glossary"],
  ];

  var here = location.pathname.split("/").pop() || "index.html";
  var nav = document.querySelector("nav.sidebar");
  if (!nav) return;

  var html =
    '<p class="brand"><a href="index.html">ELLE · DOCS</a></p>' +
    '<p class="tagline">How the AI phone, SMS and web assistant is built</p><ul>';
  for (var i = 0; i < PAGES.length; i++) {
    var active = PAGES[i][0] === here ? ' class="active"' : "";
    html += '<li><a href="' + PAGES[i][0] + '"' + active + ">" + PAGES[i][1] + "</a></li>";
  }
  html +=
    "</ul>" +
    '<div class="foot">Lewis W. Campbell<br>' +
    '<a href="https://lewiswcampbell.com">lewiswcampbell.com</a><br>' +
    '<a href="https://github.com/LewisWCampbell/ELLIE">github.com/LewisWCampbell/ELLIE</a></div>';
  nav.innerHTML = html;

  // Mobile: hamburger toggle.
  var btn = document.querySelector(".menu-btn");
  if (btn) {
    btn.addEventListener("click", function () {
      nav.classList.toggle("open");
    });
    document.addEventListener("click", function (e) {
      if (nav.classList.contains("open") && !nav.contains(e.target) && e.target !== btn && !btn.contains(e.target)) {
        nav.classList.remove("open");
      }
    });
  }

  // Copy buttons on code blocks.
  document.querySelectorAll(".codeblock").forEach(function (block) {
    var bar = block.querySelector(".src");
    var pre = block.querySelector("pre");
    if (!bar || !pre) return;
    var b = document.createElement("button");
    b.textContent = "copy";
    b.style.cssText =
      "background:none;border:1px solid var(--line);border-radius:5px;color:inherit;" +
      "font:inherit;padding:2px 8px;cursor:pointer";
    b.addEventListener("click", function () {
      navigator.clipboard.writeText(pre.textContent).then(function () {
        b.textContent = "copied";
        setTimeout(function () { b.textContent = "copy"; }, 1200);
      });
    });
    bar.appendChild(b);
  });
})();
