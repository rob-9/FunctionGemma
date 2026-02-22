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
// Cycle: offline → weak signal → connected → back to offline
// Each bar animates individually to emphasize the sequential buildup/teardown
const signalStates = [
    { bars: 0, color: '', text: 'No signal — running on-device' },
    { bars: 1, color: '#c4956a', text: 'Weak signal detected' },
    { bars: 2, color: '#c4956a', text: 'Signal strengthening...' },
    { bars: 4, color: '#6b9e78', text: 'Connected — syncing with cloud' },
    { bars: 4, color: '#6b9e78', text: 'Synced' },
    { bars: 2, color: '#c4956a', text: 'Signal fading...' },
];

let signalIdx = 0;
const dim = 'rgba(240,237,230,0.15)';
const bars = document.querySelectorAll('#signal-bars [data-bar]');
const signalText = document.getElementById('signal-text');

function cycleSignal() {
    const state = signalStates[signalIdx];

    // Animate each bar with a stagger
    bars.forEach((bar, i) => {
        setTimeout(() => {
            bar.style.backgroundColor = i < state.bars ? state.color : dim;
        }, i * 80);
    });

    // Fade the text out, swap, fade back in
    signalText.style.opacity = '0';
    setTimeout(() => {
        signalText.textContent = state.text;
        signalText.style.opacity = '1';
    }, 250);

    signalIdx = (signalIdx + 1) % signalStates.length;
}

signalText.style.transition = 'opacity 0.25s ease';
cycleSignal();
setInterval(cycleSignal, 2500);

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
//   0 — Single tool, HIGH confidence  -> on-device
//   1 — Single tool, LOW confidence + OFFLINE -> provisional, queued
//   2 — Multi-tool, HIGH confidence -> on-device (simple well-known tools)
//   3 — Multi-tool, LOW confidence + ONLINE -> cloud fallback
//   4 — Special: switch to online and drain the pending queue
const examples = [
    {
        // Single tool -> high confidence, on-device, instant
        query: 'Set an alarm for 6 AM',
        localConfidence: 0.97,
        localTimeMs: 48,
        localCalls: [{ name: 'set_alarm', args: { hour: 6, minute: 0 } }],
        cloudTimeMs: 310,
        cloudCalls: [{ name: 'set_alarm', args: { hour: 6, minute: 0 } }],
        forceConnectivity: null,  // works either way — high conf means always local
    },
    {
        // Single tool -> low confidence + offline -> queued for verification
        // Ambiguous message content — model unsure about phrasing
        query: "Text Sarah saying I found the trailhead",
        localConfidence: 0.55,
        localTimeMs: 58,
        localCalls: [{ name: 'send_message', args: { recipient: 'Sarah', message: 'found the trailhead' } }],
        cloudTimeMs: 310,
        cloudCalls: [{ name: 'send_message', args: { recipient: 'Sarah', message: 'I found the trailhead' } }],
        forceConnectivity: false,  // force offline — queued for later
    },
    {
        // Multi-tool -> high confidence, on-device, works offline
        // Simple well-known tools — local model handles all three
        query: 'Set a 30 min timer, play jazz, and remind me to stretch at 4 PM',
        localConfidence: 0.94,
        localTimeMs: 120,
        localCalls: [
            { name: 'set_timer', args: { minutes: 30 } },
            { name: 'play_music', args: { song: 'jazz' } },
            { name: 'create_reminder', args: { title: 'stretch', time: '4:00 PM' } },
        ],
        cloudTimeMs: 420,
        cloudCalls: [
            { name: 'set_timer', args: { minutes: 30 } },
            { name: 'play_music', args: { song: 'jazz' } },
            { name: 'create_reminder', args: { title: 'stretch', time: '4:00 PM' } },
        ],
        forceConnectivity: null,  // high conf — works either way
    },
    {
        // Multi-tool + online -> low confidence, cloud fallback
        // Same type of query but with signal — cloud gets all 3
        query: 'Find Bob in contacts, message him happy birthday, and set an alarm for 8 AM',
        localConfidence: 0.28,
        localTimeMs: 190,
        localCalls: [
            { name: 'search_contacts', args: { query: 'Bob' } },
            { name: 'send_message', args: { recipient: 'Bob', message: 'happy birthday' } },
        ],
        cloudTimeMs: 410,
        cloudCalls: [
            { name: 'search_contacts', args: { query: 'Bob' } },
            { name: 'send_message', args: { recipient: 'Bob', message: 'happy birthday' } },
            { name: 'set_alarm', args: { hour: 8, minute: 0 } },
        ],
        forceConnectivity: true,  // force online — cloud fallback
    },
    {
        // Special — drain the queue
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

    // Handle the special drain-queue action
    if (ex.special === 'drain-queue') {
        runDrainDemo();
        return;
    }

    // Force connectivity state if the example specifies it
    if (ex.forceConnectivity !== null && ex.forceConnectivity !== undefined) {
        setConnectivity(ex.forceConnectivity);
    }

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
        uxMessage = 'Low confidence — queued for cloud verification';
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
                    Cloud returned ${ex.cloudCalls.length} calls vs ${ex.localCalls.length} from local model — local model dropped a call
                </div>
            `);
            content.appendChild(diff);
        }
    }, 2100 + cloudDelay));

    // Phase 7: "Processed" confirmation
    pendingTimeouts.push(setTimeout(() => {
        content.appendChild(el('div', 'demo-line mt-4', `
            <div class="flex items-center gap-2 text-xs text-forest-400 font-mono">
                <svg class="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M20 6L9 17l-5-5"/></svg>
                Processed
            </div>
        `));
    }, 2700 + cloudDelay));

    // Phase 8: queue the item if offline + low confidence
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

// ── Drain-queue demo ──
// Switches to online and drains any pending items. If the queue is empty,
// shows a hint to try the offline-queue example first.
function runDrainDemo() {
    const placeholder = document.getElementById('demo-placeholder');
    const content = document.getElementById('demo-content');

    placeholder.classList.add('hidden');
    content.classList.remove('hidden');
    content.innerHTML = '';

    if (pendingQueue.length === 0) {
        const hint = el('div', 'demo-line', `
            <div class="flex flex-col items-center justify-center h-[280px] text-earth-50/40 text-center">
                <p class="text-sm mb-2">The queue is empty.</p>
                <p class="text-xs text-earth-50/25">Try the <strong class="text-earth-400/60">offline queue</strong> example first to add items, then come back here.</p>
            </div>
        `);
        content.appendChild(hint);
        return;
    }

    // Show the "going online" message
    const onlineLine = el('div', 'demo-line', `
        <div class="text-earth-50/40 text-xs mb-3 font-mono">
            <span class="text-forest-400">></span> connectivity restored
        </div>
        <p class="text-lg text-earth-50/90">Signal found — draining verification queue</p>
    `);
    content.appendChild(onlineLine);

    // Switch to online after a short beat
    pendingTimeouts.push(setTimeout(() => {
        setConnectivity(true);
    }, 600));

    // Show syncing spinner
    pendingTimeouts.push(setTimeout(() => {
        const syncSpinner = el('div', 'demo-line mt-5', `
            <div class="flex items-center gap-3 text-sm text-earth-50/50">
                <svg class="w-4 h-4 spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/>
                </svg>
                <span>Verifying ${pendingQueue.length} queued ${pendingQueue.length === 1 ? 'query' : 'queries'} via Gemini 2.5 Flash...</span>
            </div>
        `);
        content.appendChild(syncSpinner);
    }, 900));

    // Show completion
    const queueLen = pendingQueue.length;
    pendingTimeouts.push(setTimeout(() => {
        const doneLine = el('div', 'demo-line mt-5', `
            <div class="route-badge">
                <span class="route-dot on-device"></span>
                <span class="text-forest-400">Queue drained</span>
            </div>
            <div class="text-xs text-earth-50/30 mt-1">
                ${queueLen} ${queueLen === 1 ? 'answer' : 'answers'} verified against cloud — corrections surfaced
            </div>
        `);
        content.appendChild(doneLine);
    }, 2200));
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

// ── Chat form handler ──
const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');

if (chatForm) {
    chatForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const query = chatInput.value.trim();
        if (!query) return;

        // Clear active chip
        document.querySelectorAll('.demo-chip').forEach(c => c.classList.remove('active'));

        // Run as a custom query with simulated routing
        runCustomQuery(query);
        chatInput.value = '';
    });
}

function runCustomQuery(query) {
    clearPending();

    const placeholder = document.getElementById('demo-placeholder');
    const content = document.getElementById('demo-content');

    placeholder.classList.add('hidden');
    content.classList.remove('hidden');
    content.innerHTML = '';

    // Simulate: high confidence if short/simple, low if long/complex
    const wordCount = query.split(/\s+/).length;
    const hasMultiIntent = /\band\b|,/.test(query);
    const conf = hasMultiIntent ? 0.35 : (wordCount <= 6 ? 0.93 : 0.72);
    const level = confLevel(conf);
    const pct = Math.round(conf * 100);

    const route = level === 'high' ? 'on-device' : (isOnline ? 'cloud' : 'queued');
    const timeMs = route === 'on-device' ? 45 : (route === 'cloud' ? 390 : 55);

    // Phase 1: query
    content.appendChild(el('div', 'demo-line', `
        <div class="text-earth-50/40 text-xs mb-3 font-mono">
            <span class="text-earth-400">></span> query
        </div>
        <p class="text-lg text-earth-50/90">"${query}"</p>
    `));

    // Phase 2: processing
    pendingTimeouts.push(setTimeout(() => {
        content.appendChild(el('div', 'demo-line mt-5', `
            <div class="flex items-center gap-3 text-sm text-earth-50/50">
                <svg class="w-4 h-4 spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/>
                </svg>
                <span>Processing on-device...</span>
            </div>
        `));
    }, 300));

    // Phase 3: confidence
    pendingTimeouts.push(setTimeout(() => {
        content.appendChild(el('div', 'demo-line mt-5', `
            <div class="flex items-center justify-between text-xs mb-2">
                <span class="text-earth-50/40 font-mono">local confidence</span>
                <span class="text-earth-50/60">${pct}%</span>
            </div>
            <div class="confidence-track">
                <div class="confidence-fill ${level}" id="conf-bar"></div>
            </div>
        `));
        requestAnimationFrame(() => {
            const bar = document.getElementById('conf-bar');
            if (bar) bar.style.width = pct + '%';
        });
    }, 900));

    // Phase 4: routing
    pendingTimeouts.push(setTimeout(() => {
        let routeHtml;
        if (route === 'on-device') {
            routeHtml = `<div class="route-badge"><span class="route-dot on-device"></span><span class="text-forest-400">On-device</span><span class="text-earth-50/30 text-xs ml-1">${timeMs}ms</span></div>`;
        } else if (route === 'cloud') {
            routeHtml = `<div class="route-badge"><span class="route-dot cloud"></span><span class="text-earth-400">Cloud fallback</span><span class="text-earth-50/30 text-xs ml-1">${timeMs}ms</span></div>
                <div class="text-xs text-earth-50/30 mt-1">Routed to Gemini 2.5 Flash</div>`;
        } else {
            routeHtml = `<div class="route-badge"><span class="route-dot queued"></span><span class="text-amber-400/80">On-device (provisional)</span><span class="text-earth-50/30 text-xs ml-1">${timeMs}ms</span></div>
                <div class="text-xs text-earth-50/30 mt-1">Low confidence — queued for cloud verification</div>`;
        }
        content.appendChild(el('div', 'demo-line mt-4', routeHtml));
    }, 1500));

    // Phase 5: result note
    pendingTimeouts.push(setTimeout(() => {
        content.appendChild(el('div', 'demo-line mt-6 pt-5 border-t border-earth-50/[0.06]', `
            <div class="text-xs text-earth-50/25 font-mono">Connect a backend to see real function calls</div>
        `));
    }, 2100));

    // Queue if offline + low
    if (route === 'queued') {
        pendingTimeouts.push(setTimeout(() => {
            pendingQueue.push({ query, confidence: conf, resolved: false });
            renderQueue();
        }, 2600));
    }
}
