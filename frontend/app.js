// ── State ──
let isOnline = false;
let pendingQueue = [];
let pendingTimeouts = [];

// ── Navigation scroll effect ──
const nav = document.getElementById('nav');
window.addEventListener('scroll', () => {
    nav.classList.toggle('nav-scrolled', window.scrollY > 50);
});

// ── Signal indicator animation ──
const signalStates = [
    { bars: 0, color: '', text: 'No signal — running on-device' },
    { bars: 1, color: '#c4956a', text: 'Weak signal detected' },
    { bars: 4, color: '#6b9e78', text: 'Connected — syncing with cloud' },
];

let signalIdx = 0;
const dim = 'rgba(240,237,230,0.15)';

function cycleSignal() {
    const state = signalStates[signalIdx];
    document.querySelectorAll('#signal-bars [data-bar]').forEach((bar, i) => {
        bar.style.backgroundColor = i < state.bars ? state.color : dim;
    });
    document.getElementById('signal-text').textContent = state.text;
    signalIdx = (signalIdx + 1) % signalStates.length;
}

cycleSignal();
setInterval(cycleSignal, 3000);

// ── Scroll reveal ──
const revealObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('visible');
            revealObserver.unobserve(entry.target);
        }
    });
}, { threshold: 0.15 });

document.querySelectorAll('[data-reveal]').forEach(el => revealObserver.observe(el));

// ── Connectivity toggle ──
const btnOffline = document.getElementById('toggle-offline');
const btnOnline = document.getElementById('toggle-online');
const badge = document.getElementById('connectivity-badge');

function setConnectivity(online) {
    isOnline = online;

    btnOffline.classList.toggle('active', !online);
    btnOffline.classList.toggle('offline', !online);
    btnOnline.classList.toggle('active', online);
    btnOnline.classList.toggle('online', online);

    const dot = badge.querySelector('span:first-child');
    const label = badge.querySelector('span:last-child');

    if (online) {
        dot.className = 'w-2 h-2 rounded-full bg-forest-400';
        label.textContent = 'online';
    } else {
        dot.className = 'w-2 h-2 rounded-full bg-red-400/80';
        label.textContent = 'offline';
    }

    // If switching to online, drain the queue
    if (online && pendingQueue.length > 0) {
        drainQueue();
    }
}

btnOffline.addEventListener('click', () => setConnectivity(false));
btnOnline.addEventListener('click', () => setConnectivity(true));

// ── Demo examples ──
// Each example forces a specific connectivity state to clearly demonstrate
// one routing path. The 'forceConnectivity' field sets online/offline before
// running, so every chip click produces a deterministic result.
//
// Routing paths demonstrated:
//   0 — HIGH confidence  -> on-device (works offline or online)
//   1 — LOW confidence + OFFLINE -> provisional answer, queued for later
//   2 — LOW confidence + ONLINE  -> cloud fallback via Gemini
//   3 — MULTI-TOOL query + ONLINE -> cloud (local model struggles with multi-call)
//   4 — Special: switch to online and drain the pending queue
const examples = [
    {
        // Path 1: High confidence -> on-device, instant
        query: 'Set an alarm for 6 AM',
        localConfidence: 0.97,
        localTimeMs: 48,
        localCalls: [{ name: 'set_alarm', args: { hour: 6, minute: 0 } }],
        cloudTimeMs: 310,
        cloudCalls: [{ name: 'set_alarm', args: { hour: 6, minute: 0 } }],
        forceConnectivity: null,  // works either way
    },
    {
        // Path 2: Low confidence + offline -> provisional + queued
        query: 'How cold will it get tonight?',
        localConfidence: 0.58,
        localTimeMs: 61,
        localCalls: [{ name: 'get_weather', args: { location: 'current', forecast: 'tonight' } }],
        cloudTimeMs: 350,
        cloudCalls: [{ name: 'get_weather', args: { location: 'current', forecast: 'tonight low' } }],
        forceConnectivity: false,  // force offline
    },
    {
        // Path 3: Low confidence + online -> cloud fallback
        query: 'Find the nearest water source',
        localConfidence: 0.52,
        localTimeMs: 58,
        localCalls: [{ name: 'search_nearby', args: { query: 'water' } }],
        cloudTimeMs: 380,
        cloudCalls: [{ name: 'search_nearby', args: { query: 'water source', radius_mi: 5 } }],
        forceConnectivity: true,  // force online
    },
    {
        // Path 4: Multi-tool query -> cloud needed (local drops calls)
        query: 'Check the weather, start a 30 min timer, and remind me to filter water at 5 PM',
        localConfidence: 0.31,
        localTimeMs: 185,
        localCalls: [
            { name: 'get_weather', args: { location: 'current' } },
            { name: 'set_timer', args: { minutes: 30 } },
        ],
        cloudTimeMs: 420,
        cloudCalls: [
            { name: 'get_weather', args: { location: 'current' } },
            { name: 'set_timer', args: { minutes: 30 } },
            { name: 'create_reminder', args: { title: 'filter water', time: '5:00 PM' } },
        ],
        forceConnectivity: true,  // force online to show cloud value
    },
    {
        // Path 5: Special — drain the queue (no query, just toggle online)
        special: 'drain-queue',
    },
];

// ── Formatting helpers ──

function formatCalls(calls) {
    return calls.map(call => {
        const args = Object.entries(call.args)
            .map(([k, v]) => {
                const val = typeof v === 'string'
                    ? `<span class="fn-punct">"</span><span class="fn-value">${v}</span><span class="fn-punct">"</span>`
                    : `<span class="fn-value">${v}</span>`;
                return `  <span class="fn-key">${k}</span><span class="fn-punct">:</span> ${val}`;
            })
            .join('\n');
        return `<span class="fn-name">${call.name}</span><span class="fn-punct">(</span>\n${args}\n<span class="fn-punct">)</span>`;
    }).join('\n\n');
}

function confLevel(conf) {
    if (conf >= 0.90) return 'high';
    if (conf >= 0.70) return 'medium';
    return 'low';
}

// ── Queue management ──

function renderQueue() {
    const panel = document.getElementById('queue-panel');
    const items = document.getElementById('queue-items');
    const count = document.getElementById('queue-count');

    if (pendingQueue.length === 0) {
        panel.classList.add('hidden');
        return;
    }

    panel.classList.remove('hidden');
    count.textContent = pendingQueue.length + (pendingQueue.length === 1 ? ' item' : ' items');

    items.innerHTML = pendingQueue.map(q => {
        const resolved = q.resolved;
        return `
            <div class="queue-item ${resolved ? 'resolved' : ''}">
                <div class="flex items-center gap-2">
                    <span class="route-dot ${resolved ? 'on-device' : 'queued'}"></span>
                    <span class="text-earth-50/70">${q.query}</span>
                </div>
                <span class="text-xs ${resolved ? 'text-forest-400' : 'text-earth-50/30'} font-mono">
                    ${resolved ? 'verified' : 'pending'}
                </span>
            </div>
        `;
    }).join('');

    lucide.createIcons();
}

function drainQueue() {
    let delay = 0;
    pendingQueue.forEach((item, i) => {
        if (item.resolved) return;
        delay += 600;
        pendingTimeouts.push(setTimeout(() => {
            item.resolved = true;
            renderQueue();
        }, delay));
    });

    // Clear resolved items after showing them
    pendingTimeouts.push(setTimeout(() => {
        pendingQueue = pendingQueue.filter(q => !q.resolved);
        renderQueue();
    }, delay + 2000));
}

// ── Demo runner ──

function clearPending() {
    pendingTimeouts.forEach(clearTimeout);
    pendingTimeouts = [];
}

function runDemo(idx) {
    clearPending();

    const ex = examples[idx];
    const conf = ex.localConfidence;
    const level = confLevel(conf);
    const pct = Math.round(conf * 100);

    const placeholder = document.getElementById('demo-placeholder');
    const content = document.getElementById('demo-content');

    placeholder.classList.add('hidden');
    content.classList.remove('hidden');
    content.innerHTML = '';

    // Determine routing based on confidence + connectivity
    let route, finalCalls, finalTimeMs, uxMessage;

    if (level === 'high') {
        // High confidence: always local, regardless of connectivity
        route = 'on-device';
        finalCalls = ex.localCalls;
        finalTimeMs = ex.localTimeMs;
        uxMessage = null;
    } else if (isOnline) {
        // Low/medium confidence + online: cloud fallback
        route = 'cloud';
        finalCalls = ex.cloudCalls;
        finalTimeMs = ex.localTimeMs + ex.cloudTimeMs;
        uxMessage = null;
    } else {
        // Low/medium confidence + offline: local result + queue
        route = 'queued';
        finalCalls = ex.localCalls;
        finalTimeMs = ex.localTimeMs;
        uxMessage = level === 'medium'
            ? 'Verify when you have signal'
            : 'Low confidence — queued for cloud verification';
    }

    // Phase 1: query
    const queryLine = el('div', 'demo-line', `
        <div class="text-earth-50/40 text-xs mb-3 font-mono">
            <span class="text-earth-400">></span> query
        </div>
        <p class="text-lg text-earth-50/90">"${ex.query}"</p>
    `);
    content.appendChild(queryLine);

    // Phase 2: processing spinner
    pendingTimeouts.push(setTimeout(() => {
        const processing = el('div', 'demo-line mt-5', `
            <div class="flex items-center gap-3 text-sm text-earth-50/50" id="demo-processing">
                <svg class="w-4 h-4 spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/>
                </svg>
                <span>Processing on-device...</span>
            </div>
        `);
        content.appendChild(processing);
    }, 300));

    // Phase 3: confidence result
    pendingTimeouts.push(setTimeout(() => {
        const confBlock = el('div', 'demo-line mt-5', `
            <div class="flex items-center justify-between text-xs mb-2">
                <span class="text-earth-50/40 font-mono">local confidence</span>
                <span class="text-earth-50/60">${pct}%</span>
            </div>
            <div class="confidence-track">
                <div class="confidence-fill ${level}" id="conf-bar"></div>
            </div>
        `);
        content.appendChild(confBlock);

        requestAnimationFrame(() => {
            const bar = document.getElementById('conf-bar');
            if (bar) bar.style.width = pct + '%';
        });
    }, 900));

    // Phase 4: routing decision
    pendingTimeouts.push(setTimeout(() => {
        let routeHtml;

        if (route === 'on-device') {
            routeHtml = `
                <div class="route-badge">
                    <span class="route-dot on-device"></span>
                    <span class="text-forest-400">On-device</span>
                    <span class="text-earth-50/30 text-xs ml-1">${finalTimeMs}ms</span>
                </div>
            `;
        } else if (route === 'cloud') {
            routeHtml = `
                <div class="route-badge">
                    <span class="route-dot cloud"></span>
                    <span class="text-earth-400">Cloud fallback</span>
                    <span class="text-earth-50/30 text-xs ml-1">${finalTimeMs}ms</span>
                </div>
                <div class="text-xs text-earth-50/30 mt-1">
                    Local confidence too low — routed to Gemini 2.5 Flash
                </div>
            `;
        } else {
            // queued
            routeHtml = `
                <div class="route-badge">
                    <span class="route-dot queued"></span>
                    <span class="text-amber-400/80">On-device (provisional)</span>
                    <span class="text-earth-50/30 text-xs ml-1">${finalTimeMs}ms</span>
                </div>
                <div class="text-xs text-earth-50/30 mt-1">
                    ${uxMessage}
                </div>
            `;
        }

        const routeBlock = el('div', 'demo-line mt-4', routeHtml);
        content.appendChild(routeBlock);
    }, 1500));

    // Phase 5: cloud processing (if cloud route)
    const cloudDelay = route === 'cloud' ? 800 : 0;

    if (route === 'cloud') {
        pendingTimeouts.push(setTimeout(() => {
            const cloudSpinner = el('div', 'demo-line mt-4', `
                <div class="flex items-center gap-3 text-sm text-earth-50/50">
                    <svg class="w-4 h-4 spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/>
                    </svg>
                    <span>Querying Gemini 2.5 Flash...</span>
                </div>
            `);
            content.appendChild(cloudSpinner);
        }, 2000));
    }

    // Phase 6: function calls
    pendingTimeouts.push(setTimeout(() => {
        const fnBlock = el('div', 'demo-line mt-6 pt-5 border-t border-earth-50/[0.06]', `
            <div class="text-xs text-earth-50/30 font-mono mb-3">function calls</div>
            <pre class="fn-call bg-earth-900/50 rounded-lg p-4 overflow-x-auto">${formatCalls(finalCalls)}</pre>
        `);
        content.appendChild(fnBlock);

        // If this was a multi-tool cloud query, highlight the difference
        if (route === 'cloud' && ex.cloudCalls.length > ex.localCalls.length) {
            const diff = el('div', 'demo-line mt-3', `
                <div class="text-xs text-earth-400">
                    Cloud returned ${ex.cloudCalls.length} calls vs ${ex.localCalls.length} from local model
                </div>
            `);
            content.appendChild(diff);
        }
    }, 2100 + cloudDelay));

    // Phase 7: queue the item if offline + low confidence
    if (route === 'queued') {
        pendingTimeouts.push(setTimeout(() => {
            pendingQueue.push({
                query: ex.query,
                confidence: conf,
                resolved: false,
            });
            renderQueue();
        }, 2600));
    }
}

// ── Helper ──
function el(tag, className, innerHTML) {
    const node = document.createElement(tag);
    node.className = className;
    node.innerHTML = innerHTML;
    return node;
}

// ── Chip click handlers ──
document.querySelectorAll('.demo-chip').forEach(chip => {
    chip.addEventListener('click', () => {
        document.querySelectorAll('.demo-chip').forEach(c => c.classList.remove('active'));
        chip.classList.add('active');
        runDemo(parseInt(chip.dataset.example));
    });
});
