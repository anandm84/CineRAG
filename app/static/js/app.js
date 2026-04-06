// CineRAG Frontend — vanilla JS

const queryInput = document.getElementById("query-input");
const searchBtn = document.getElementById("search-btn");
const loadingEl = document.getElementById("loading");
const errorEl = document.getElementById("error");
const resultsEl = document.getElementById("results");
const answerText = document.getElementById("answer-text");
const sourcesList = document.getElementById("sources-list");
const queryTypeBadge = document.getElementById("query-type-badge");
const resultMeta = document.getElementById("result-meta");

// Filters
const filterLang = document.getElementById("filter-lang");
const filterSource = document.getElementById("filter-source");
const filterYearMin = document.getElementById("filter-year-min");
const filterYearMax = document.getElementById("filter-year-max");

// Search handler
function doSearch() {
    const question = queryInput.value.trim();
    if (!question) return;

    // Gather filters
    const payload = { question };
    if (filterLang && filterLang.value) payload.language = filterLang.value;
    if (filterSource && filterSource.value) payload.source_type = filterSource.value;
    if (filterYearMin && filterYearMin.value) payload.year_min = filterYearMin.value;
    if (filterYearMax && filterYearMax.value) payload.year_max = filterYearMax.value;

    // UI state
    loadingEl.classList.remove("hidden");
    errorEl.classList.add("hidden");
    resultsEl.classList.add("hidden");
    searchBtn.disabled = true;

    const startTime = Date.now();

    fetch("/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    })
        .then((res) => {
            if (!res.ok) return res.json().then((d) => Promise.reject(d));
            return res.json();
        })
        .then((data) => {
            const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
            renderResults(data, elapsed);
        })
        .catch((err) => {
            const msg = err.error || err.message || "Something went wrong";
            errorEl.textContent = msg;
            errorEl.classList.remove("hidden");
        })
        .finally(() => {
            loadingEl.classList.add("hidden");
            searchBtn.disabled = false;
        });
}

// Render results
function renderResults(data, elapsed) {
    // Badge
    const qtype = data.query_type || "general";
    queryTypeBadge.textContent = qtype.replace("_", "/");
    queryTypeBadge.className = "badge badge-" + qtype;

    // Meta
    const sourceCount = (data.sources || []).length;
    resultMeta.textContent = `${sourceCount} sources \u00b7 ${elapsed}s`;

    // Answer
    answerText.textContent = data.answer || "No answer generated.";

    // Sources
    sourcesList.innerHTML = "";
    (data.sources || []).forEach((src) => {
        const card = document.createElement("div");
        card.className = "source-card";
        card.innerHTML = `
            <div class="source-header">
                <span class="source-movie">${esc(src.movie)}</span>
                <span class="source-year">(${src.year})</span>
                <span class="source-type">${esc(src.source_type)}</span>
                ${src.language ? `<span class="source-type">${esc(src.language)}</span>` : ""}
            </div>
            <div class="source-snippet">${esc(src.snippet)}</div>
        `;
        sourcesList.appendChild(card);
    });

    resultsEl.classList.remove("hidden");
}

// Escape HTML
function esc(str) {
    if (!str) return "";
    const d = document.createElement("div");
    d.textContent = str;
    return d.innerHTML;
}

// Event listeners
if (searchBtn) {
    searchBtn.addEventListener("click", doSearch);
}

if (queryInput) {
    queryInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") doSearch();
    });
}

// Example query buttons
document.querySelectorAll(".example").forEach((btn) => {
    btn.addEventListener("click", () => {
        queryInput.value = btn.dataset.q;
        doSearch();
    });
});

// Stats page — render if we're on /stats
if (window.location.pathname === "/stats") {
    fetch("/api/stats")
        .then((res) => res.json())
        .then((data) => renderStats(data))
        .catch((err) => {
            document.querySelector(".container").innerHTML =
                `<div class="error-box">Failed to load stats: ${err.message}</div>`;
        });
}

function renderStats(data) {
    const container = document.querySelector(".container");
    if (!container) return;

    let html = '<h1 style="margin-bottom: 8px;">Stats</h1>';
    html += '<p style="color: var(--text-dim); margin-bottom: 24px;">Database and index statistics</p>';
    html += '<div class="stats-grid">';

    // Database stats
    const db = data.database || {};
    if (!db.error) {
        html += `
            <div class="stat-card">
                <h3>Total Movies</h3>
                <div class="stat-value">${db.total_movies || 0}</div>
            </div>
            <div class="stat-card">
                <h3>By Language</h3>
                <ul class="stat-list">
                    ${Object.entries(db.by_language || {})
                        .map(([k, v]) => `<li><span>${langName(k)}</span><span>${v}</span></li>`)
                        .join("")}
                </ul>
            </div>
            <div class="stat-card">
                <h3>Classification</h3>
                <ul class="stat-list">
                    <li><span>Hits</span><span>${db.hits || 0}</span></li>
                    <li><span>Flops</span><span>${db.flops || 0}</span></li>
                </ul>
            </div>
            <div class="stat-card">
                <h3>Data Availability</h3>
                <ul class="stat-list">
                    <li><span>With Scripts</span><span>${db.with_scripts || 0}</span></li>
                    <li><span>With Reviews</span><span>${db.with_reviews || 0}</span></li>
                </ul>
            </div>
        `;
    }

    // Vector store stats
    const vs = data.vector_store || {};
    if (!vs.error) {
        html += `
            <div class="stat-card">
                <h3>Vector Index</h3>
                <div class="stat-value">${vs.total_chunks || 0}</div>
                <p style="color: var(--text-dim); font-size: 0.85rem; margin-top: 4px;">chunks indexed</p>
            </div>
        `;
    }

    // Eval metrics
    const ev = data.evaluation || {};
    if (ev.total_questions) {
        html += `
            <div class="stat-card">
                <h3>Evaluation</h3>
                <ul class="stat-list">
                    <li><span>Questions</span><span>${ev.total_questions}</span></li>
                    <li><span>Query Type Accuracy</span><span>${(ev.query_type_accuracy * 100).toFixed(0)}%</span></li>
                    <li><span>Source Hit Rate</span><span>${(ev.source_type_hit_rate * 100).toFixed(0)}%</span></li>
                    <li><span>Movie Hit Rate</span><span>${(ev.movie_hit_rate * 100).toFixed(0)}%</span></li>
                    <li><span>Avg Response Time</span><span>${ev.avg_response_time_s}s</span></li>
                </ul>
            </div>
        `;

        if (ev.ragas) {
            html += `
                <div class="stat-card">
                    <h3>RAGAS Scores</h3>
                    <ul class="stat-list">
                        ${Object.entries(ev.ragas)
                            .map(([k, v]) => `<li><span>${k}</span><span>${v.toFixed(4)}</span></li>`)
                            .join("")}
                    </ul>
                </div>
            `;
        }
    }

    html += "</div>";
    container.innerHTML = html;
}

function langName(code) {
    const map = { te: "Telugu", hi: "Hindi", en: "English" };
    return map[code] || code;
}
